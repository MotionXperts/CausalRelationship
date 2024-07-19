import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration
import torch
import json
from tqdm import tqdm

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

def evaluate(dataset, model, device):
    model.eval()
    val_data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
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
                                max_length=50,
                                num_beams=beam_size,
                                repetition_penalty=2.5,
                                length_penalty=1.0,
                                early_stopping=True
                            )
                    
            for name, gen_id in zip(video_names, generated_ids):
                decoded_text = tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                results[name] = decoded_text

    with open('modify_results_test_0702.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE:", DEVICE)
    parser = argparse.ArgumentParser()
    tokenizer = AutoTokenizer.from_pretrained('t5-large', use_fast=True)
    model = ModifiedT5Model()
    model.to(DEVICE)
    # Load the trained model
    # model_path = '/home/peihsin/Sinica/models/Axel_epoch64.pt'
    model_path = '/home/andrewchen/Sinica/models/emnlp_scl__epoch50.pt'
    model.load_state_dict(torch.load(model_path))
    # parser.add_argument('--test_data', default='./dataset/align_val.pkl') 
    parser.add_argument('--test_data', default='./dataset/output_test_label_para6.pkl')
    args = parser.parse_args()

    # Test the model
    test_dataset = HumanMLDataset_val(args.test_data, tokenizer)    
    print("Test dataset size:", len(test_dataset))
    evaluate(test_dataset, model, device=DEVICE)
