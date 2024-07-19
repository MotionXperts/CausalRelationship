import torch
from torch.nn import functional as nnf
from transformers import T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForSeq2Seq
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from torch.nn.utils.rnn import pad_sequence
import json
import time
import matplotlib.pyplot as plt
from cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore

class HumanMLDataset(Dataset):
    def __init__(self, pkl_file, tokenizer, transform=None):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.samples = []
        max_len = 0  
        for item in self.data_list:
            features = item['features']
            max_len = max(max_len, len(features)) 
            video_name = item['video_name']
            #video_name = item['video']
            for label in item['labels']:
            #for label in item['description']:
                self.samples.append((features, label, video_name))
        self.max_len = max_len  
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features, label, video_name = self.samples[idx]
        #print(f'Label: {label}')

        padded_features = np.zeros((self.max_len, 66))
        keypoints_mask = np.zeros(self.max_len) 

        current_len = len(features)
        padded_features[:current_len, :] = features
        keypoints_mask[:current_len] = 1  
        tokenized_label = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        start_token_id = 0
        tokenized_label['input_ids'] = torch.cat((torch.tensor([[start_token_id]]), tokenized_label['input_ids']), 1)
        #print(f'Tokenized label: {tokenized_label}')

        sample = {
            "video_name": video_name,
            "keypoints": torch.FloatTensor(padded_features),
            "keypoints_mask": torch.FloatTensor(keypoints_mask),
            "label": label,
            "output": tokenized_label['input_ids'].squeeze(0),
        }
        return sample
    
class CustomDataset(Dataset):
    def __init__(self, pkl_file, tokenizer, transform=None):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.max_len = max(len(item['features']) for item in self.data_list)
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data_list[idx]
        #features, video_name, label = item['features'], item['video_name'], item['labels']            
        #features, video_name, label = item['features'], item['video_name'], item['output'] 
        features, video_name, label = item['features'], item['video'], item['description']      
        #print(f'Label: {label}')

        padded_features = np.zeros((self.max_len, 66))
        keypoints_mask = np.zeros(self.max_len) 

        current_len = len(features)
        padded_features[:current_len, :] = features
        keypoints_mask[:current_len] = 1 
        tokenized_label = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=300)
        start_token_id = 0
        tokenized_label['input_ids'] = torch.cat((torch.tensor([[start_token_id]]), tokenized_label['input_ids']), 1)

        sample = {
            "video_name": video_name,
            "keypoints": torch.FloatTensor(padded_features),
            "keypoints_mask": torch.FloatTensor(keypoints_mask),
            "label": label,
            "output": tokenized_label['input_ids'].squeeze(0),
        }
        return sample
    
class HumanMLDataset_val(Dataset):
    def __init__(self, pkl_file, tokenizer, transform=None):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max(len(item['features']) for item in self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data_list[idx]
        #features, video_name = item['features'], item['video_name']
        features, video_name = item['features'], item['video_name']

        padded_features = np.zeros((self.max_len, 66))
        keypoints_mask = np.zeros(self.max_len) 

        current_len = len(features)
        padded_features[:current_len, :] = features
        keypoints_mask[:current_len] = 1  

        sample = {
            "video_name": video_name,
            "keypoints": torch.FloatTensor(padded_features),
            "keypoints_mask": torch.FloatTensor(keypoints_mask),
        }
        return sample

class ModifiedT5Model(nn.Module):

    def __init__(self, embed_size=384, seq_length=768, num_joints=22, num_features=3):
        super(ModifiedT5Model, self).__init__()
        config = AutoConfig.from_pretrained('google/flan-t5-base')
        self.t5 = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base', config=config)
        self.embed_size = embed_size
        self.num_features = num_features
        self.spatial_embedding = nn.Linear(num_features, embed_size)
        self.temporal_embedding = nn.Embedding(seq_length, embed_size)
        self.positional_embedding = nn.Embedding(num_joints, embed_size)
        
        self.fc = nn.Sequential(nn.Linear(num_joints * embed_size, seq_length))

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None, use_embeds=True):
        if use_embeds:
            batch_size, seq_length, feature_dim = input_ids.shape
            num_joints = feature_dim // self.num_features
            
            if feature_dim % 3 != 0:
                raise ValueError(f"feature_dim should be divisible by 3, but got {feature_dim}")
            
            input_reshaped = input_ids.view(batch_size * seq_length, num_joints, self.num_features)
            spatial_embeds = self.spatial_embedding(input_reshaped).view(batch_size, seq_length, -1)
            temporal_embeds = self.temporal_embedding(torch.arange(0, seq_length, device=input_ids.device)).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_joints, self.embed_size).reshape(batch_size, seq_length, -1)
            positional_embeds = self.positional_embedding(torch.arange(0, num_joints, device=input_ids.device)).unsqueeze(0).unsqueeze(1).expand(batch_size, seq_length, num_joints, self.embed_size).reshape(batch_size, seq_length, -1)

            combined_embeds = spatial_embeds + temporal_embeds + positional_embeds
            input_embeds = self.fc(combined_embeds.view(batch_size, seq_length, -1))
            output = self.t5(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        else:
            output = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)

        return output

def train(train_dataset, model, tokenizer, args, eval_dataset=None, lr=1e-4, warmup_steps=5000, output_dir=".", output_prefix=""):
    device = torch.device('cuda')
    print(f"Training on {device}")
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    epoch_loss_list, epoch_cider_list, epoch_bleu_score_list = [], [], []
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        loss_list = []
        for idx, batch in enumerate(train_dataloader):
            model.zero_grad()
            video_names = batch['video_name']
            src_batch = batch['keypoints'].to(device)
            keypoints_mask_batch = batch['keypoints_mask'].to(device)
            tgt_batch = batch['output'].to(device)

            tgt_input = tgt_batch[:, :-1]
            tgt_labels = tgt_batch[:, 1:]
            optimizer.zero_grad()

            outputs = model(input_ids=src_batch.contiguous(), 
                            attention_mask=keypoints_mask_batch.contiguous(), 
                            decoder_input_ids=tgt_input.contiguous(),
                            labels=tgt_labels.contiguous())

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_list.append(loss.item())
            progress.set_postfix({
                'loss': np.mean(loss_list),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
            })
            progress.update()

        epoch_loss = np.mean(loss_list)
        epoch_loss_list.append(epoch_loss)
        print(f"Epoch {epoch}: Train Loss: {np.mean(loss_list):.4f}")
        torch.save(model.state_dict(), os.path.join(output_dir, f"{output_prefix}_epoch{epoch}.pt"))  

        if eval_dataset is not None:          
            model.eval()
            val_data_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
            results = {}
            start_token_id = 0
            beam_size = 3
            with torch.no_grad():
                for batch in tqdm(val_data_loader):
                    video_names = batch['video_name']
                    src_batch = batch['keypoints'].to(device)
                    keypoints_mask_batch = batch['keypoints_mask'].to(device)
                    batch_size, seq_length, feature_dim = src_batch.shape
                    num_joints = feature_dim // model.num_features
                    input_reshaped = src_batch.view(batch_size * seq_length, num_joints, model.num_features)
                    spatial_embeds = model.spatial_embedding(input_reshaped).view(batch_size, seq_length, -1)
                    temporal_embeds = model.temporal_embedding(torch.arange(0, seq_length, device=src_batch.device)).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_joints, model.embed_size).reshape(batch_size, seq_length, -1)
                    positional_embeds = model.positional_embedding(torch.arange(0, num_joints, device=src_batch.device)).unsqueeze(0).unsqueeze(1).expand(batch_size, seq_length, num_joints, model.embed_size).reshape(batch_size, seq_length, -1)
                    combined_embeds = spatial_embeds + temporal_embeds + positional_embeds
                    input_embeds = model.fc(combined_embeds.view(batch_size, seq_length, -1))
                    decode_input_ids = torch.tensor([[start_token_id]] * src_batch.shape[0]).to(device)

                    generated_ids = model.t5.generate(
                                inputs_embeds=input_embeds,
                                attention_mask=keypoints_mask_batch,
                                decoder_input_ids=decode_input_ids, 
                                max_length=300,
                                num_beams=beam_size,
                                repetition_penalty=2.5,
                                length_penalty=1.0,
                                early_stopping=True
                            )
                    
                    for name, gen_id in zip(video_names, generated_ids):
                        decoded_text = tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        results[name] = decoded_text
            with open('emnlp_scl_'+str(epoch)+'.json', 'w') as f:
                json.dump(results, f)

            predictions = readJSON('emnlp_scl_'+str(epoch)+'.json')
            annotations = readPickle('./dataset/output_test_label_para6.pkl')
            #annotations = readPickle('./dataset/rm_test.pkl')
            gts = getGTCaptions(annotations)
            #Check predictions content is correct
            assert type(predictions) is dict
            assert set(predictions.keys()) == set(gts.keys())
            assert all([type(pred) is str for pred in predictions.values()])
            # CIDErScore
            cider_score = CIDERScore()(predictions, gts)
            bleu_score = BLEUScore()(predictions, gts)
            print(f"CIDEr: {cider_score}")
            print(f"BLEU: {bleu_score}")
            epoch_cider_list.append(cider_score)
            epoch_bleu_score_list.append(bleu_score['Bleu_1'])
        
        progress.close()
    draw_metrics(epoch_loss_list, epoch_cider_list, epoch_bleu_score_list, output_dir)

    return model


def draw_metrics(loss_list, cider_list, bleu_list, save_path):
    plt.figure()
    plt.plot(loss_list, color='tab:red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.savefig(save_path + '/loss.png')
    plt.close()

    plt.figure()
    plt.plot(cider_list, color='tab:blue', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('CIDEr Score')
    plt.title('CIDEr Score over Epochs')
    plt.savefig(save_path + '/cider.png')
    plt.close()

    plt.figure()
    plt.plot(bleu_list, color='tab:green', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score over Epochs')
    plt.savefig(save_path + '/bleu.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./dataset/output_train_label_para6.pkl')
    #parser.add_argument('--data', default='./dataset/rm_train.pkl')
    parser.add_argument('--out_dir', default='./models')
    parser.add_argument('--prefix', default='emnlp_scl/emnlp_scl_', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--test_data', default='./dataset/output_test_label_para6.pkl')
    #parser.add_argument('--test_data', default='./dataset/rm_test.pkl')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', use_fast=True)
    dataset = HumanMLDataset(args.data, tokenizer)
    #dataset = CustomDataset(args.data, tokenizer)
    eval_dataset = HumanMLDataset_val(args.test_data, tokenizer)  
    model = ModifiedT5Model()
    model.load_state_dict(torch.load('./models/best_HumanML_Flan_epoch173_902.pt'))
    #print(model)

    train(dataset, model, tokenizer, args,eval_dataset=eval_dataset, output_dir=args.out_dir, output_prefix=args.prefix)

if __name__ == '__main__':
    main()