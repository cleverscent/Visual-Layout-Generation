import random
import json
import pickle
from logger_set import LOG

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from functools import partial
from data_loaders.cluster import cluster

class CanvaLayout(Dataset):
    def __init__(self, json_path, clip_json_path, text_json_path, 
                 cluster_image_model, cluster_text_model,
                 max_num_com: int , scaling_size, z_scaling_size, mean_0):
        # self.presentation_size = presentation_size
        self.max_num_element = max_num_com
        self.scaling_size = scaling_size 
        self.z_scaling_size = z_scaling_size
        self.mean_0 = mean_0
        self.data = self.process(json_path, clip_json_path, text_json_path, 
                                 cluster_image_model, cluster_text_model)
    
    def normalize_geometry(self, slide, content, num_elements):
        # Normalize left, top, width, height

        left = content['left']
        top = content['top']
        width = content['width']
        height = content['height']
        
        xc = left + width / 2.
        yc = top + height / 2.
        
        x_scale = 1920.0 / self.scaling_size
        y_scale = 1080.0 / self.scaling_size
        w_scale = 1920.0 / self.scaling_size
        h_scale = 1080.0 / self.scaling_size
        z_scale = 20.0/ self.z_scaling_size # max 요소 20개
        
        if self.mean_0 == True:
            x = xc / x_scale *2 -self.scaling_size
            y = yc / y_scale *2 -self.scaling_size
            w = content['img_width'] / w_scale *2 - self.scaling_size
            h = content['img_height'] / h_scale*2 - self.scaling_size

        else:
            x = xc / x_scale
            y = yc / y_scale            
            w = content['img_width'] / w_scale
            h = content['img_height'] / h_scale
            
        # Normalize rotation (optional)
        r = content['rotation'] / 360.0  # Assuming rotation is in degrees

        # Normalize z_index for each slide separately
        z = content['z_index'] / z_scale # max num comp로 하든말든
        return [x, y, w, h, r, z]
    

    def process(self, json_path, clip_json_path, text_json_path, 
                cluster_image_model, cluster_text_model):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(clip_json_path, 'r', encoding='utf-8') as f:
            clip_data = json.load(f)
            
        with open(text_json_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
            
        LOG.info("BEGIN: fitting clustering model on dataset")
        clustered_images = cluster(cluster_image_model, clip_json_path)
        clustered_texts = cluster(cluster_text_model, text_json_path)
        LOG.info("END: fitting clustering model on dataset")

        
        processed_data = {
                          "geometry": [], 
                          "image_features": [], 
                          "text_features": [], 
                          "ids": [], 
                          "type": [],
                          "cluster_image_id":[],
                          "cluster_text_id":[]
                        } 
        
        for presentation in data['presentations']:
            for slide in presentation['slides']:
                
                slide_geometry = []
                slide_image_features = []
                slide_text_features = []
                slide_ids = []  # 슬라이드별 id 저장을 위한 리스트
                slide_type = []
                slide_cluster_image_id = []
                slide_cluster_text_id = []
                
                num_elements = len(slide['contents'])  # Number of elements in the current slide
                
                # 여기서 요소 수가 max_num_element개를 초과하는 경우 해당 슬라이드를 무시합니다.
                if num_elements > self.max_num_element:
                    continue  # 이 슬라이드를 건너뛰고 다음 슬라이드로 넘어갑니다.
                
                # 임시 리스트에 요소들을 튜플로 저장
                elements = []
                for content in slide['contents']:
                    geometry = self.normalize_geometry(slide, content, num_elements)
                    image_file_name = content.get('image_file_name', '')
                    ppt_name = presentation['ppt_name']
                    image_features = clip_data.get(ppt_name, {}).get(image_file_name, [])
                    text_features = text_data.get(ppt_name, {}).get(image_file_name, [])
                    content_id = f"{ppt_name}/{image_file_name}"
                    content_type = content.get('type', '')
                    cluster_image_id = clustered_images.get(ppt_name, {}).get(image_file_name, [])
                    cluster_text_id = clustered_texts.get(ppt_name, {}).get(image_file_name, [])
                    
                    # 각 요소를 튜플로 묶어서 추가
                    elements.append((geometry, image_features, text_features, content_id, content_type, cluster_image_id, cluster_text_id))
                
                # elements 리스트를 무작위로 섞음
                random.shuffle(elements)
                
                # 섞인 요소들을 다시 각각의 리스트에 추가
                for geometry, image_features, text_features, content_id, content_type, cluster_image_id, cluster_text_id in elements:
                    slide_geometry.append(geometry)
                    slide_image_features.append(image_features)
                    slide_text_features.append(text_features)
                    slide_ids.append(content_id)
                    slide_type.append(content_type)
                    slide_cluster_image_id.append(cluster_image_id)
                    slide_cluster_text_id.append(cluster_text_id)
                    
                processed_data["geometry"].append(slide_geometry)
                processed_data["image_features"].append(slide_image_features)
                processed_data["text_features"].append(slide_text_features)
                processed_data["ids"].append(slide_ids)  # 슬라이드별 id 정보 추가
                processed_data["type"].append(slide_type)
                processed_data["cluster_image_id"].append(slide_cluster_image_id)
                processed_data["cluster_text_id"].append(slide_cluster_text_id)

        return processed_data

    def get_data(self):
        return self.data

    def pad_instance(self, geometry):
        padded_geometry = np.pad(geometry, pad_width=((0,self.max_num_element - np.array(geometry).shape[0]), (0, 0)), constant_values=0.)
        return padded_geometry
    
    def pad_instance_type(self, cat):
        num_pad_elements = max(0, self.max_num_element - len(cat))
        # 1차원 배열에 대한 패딩, 배열의 끝에만 패딩을 추가
        padded_cat = np.pad(cat, pad_width=(0, num_pad_elements), constant_values=0)
        return padded_cat

    def process_data(self, idx):
        geometry = self.data['geometry'][idx]
        cat = self.data['type'][idx]
        padding_mask = np.ones(np.array(geometry).shape)
        geometry = self.pad_instance(geometry)
        cat = self.pad_instance_type(cat).reshape((-1,1))
        
        padding_mask = self.pad_instance(padding_mask)
        image_features = self.data['image_features'][idx]
        text_features = self.data['text_features'][idx]
        
        padding_mask_img = np.ones(np.array(image_features).shape)
        padding_mask_img = np.squeeze(padding_mask_img, axis=1)
        padding_mask_img = self.pad_instance(padding_mask_img)
        
        padding_mask_text = np.ones(np.array(text_features).shape)
        padding_mask_text = np.squeeze(padding_mask_text, axis=1)
        padding_mask_text = self.pad_instance(padding_mask_text)
        
        image_features = np.squeeze(image_features, axis=1)
        image_features = self.pad_instance(image_features)
        
        text_features = np.squeeze(text_features, axis=1)
        text_features = self.pad_instance(text_features)
        
        cluster_image_id = self.data['cluster_image_id'][idx]
        cluster_image_id = self.pad_instance_type(cluster_image_id).reshape((-1,))
        # print("cluster_image_id:", cluster_image_id)
        
        cluster_text_id = self.data['cluster_text_id'][idx]
        cluster_text_id = self.pad_instance_type(cluster_text_id).reshape((-1,))
        
        ids = self.data['ids'][idx]  # id 정보 로드
        
        cat[cat=='freeform']=1
        cat[cat=='group']=1
        cat[cat=='picture']=2
        cat[cat=='table']=2
        cat[cat=='media']=2
        cat[cat=='auto_shape']=1
        cat[cat=='text_box']=3
        cat[cat=='0'] = 0
    
        cat = cat.reshape((-1,))
        # print("cat:", cat)
        
        return {
            "geometry": np.array(geometry).astype(np.float32),
            "image_features": np.array(image_features).astype(np.float32),
            "text_features": np.array(text_features).astype(np.float32),
            "padding_mask": padding_mask.astype(np.int32),
            "padding_mask_img": padding_mask_img.astype(np.int32),
            "padding_mask_text": padding_mask_text.astype(np.int32),
            "ids": ids,  # id 정보 반환
            "cat": cat.astype(int),
            "cluster_image_id":cluster_image_id.astype(int),
            "cluster_text_id":cluster_text_id.astype(int)
        }

    def __getitem__(self, idx):
        sample = self.process_data(idx)
        return sample

    def __len__(self):
        return len(self.data['geometry'])


# # Specify the paths to your JSON files
# json_path = r"D:\layout_cal\dataset\val_canva.json"
# clip_json_path = r"D:\layout_cal\dataset\val_clip.json"
# text_json_path = r"D:\layout_cal\dataset\val_text.json"
# cluster_image_model = r"D:\layout_cal\dataset\clustering_with_image_model.pkl"
# cluster_text_model = r"D:\layout_cal\dataset\clustering_with_text_model.pkl"

# # Create an instance of the CanvaLayout dataset
# canva_dataset = CanvaLayout(json_path, clip_json_path, text_json_path, 
#                             cluster_image_model, cluster_text_model,
#                             max_num_com=40, scaling_size=1, z_scaling_size=0.01, mean_0=True)

# # Print the total number of samples in the dataset
# print("Total number of samples:", len(canva_dataset))


# for sample_index in range(0,3):
#     # Get a specific sample from the dataset
#     sample = canva_dataset[sample_index]

#     # Print the processed sample data
#     print("\nSample data:")
#     print("Geometry:", sample["geometry"])
#     print("Image Features:", sample["image_features"])
#     print("Text Features:", sample["text_features"])
#     print("Padding Mask:", sample["padding_mask"])
#     print("Padding Mask Image:", sample["padding_mask_img"])
#     print("Padding Mask Text:", sample["padding_mask_text"])
#     print("IDs:", sample["ids"])
#     print("Type:", sample["cat"])
#     print("cluster_image_id:", sample['cluster_image_id'])
#     print("cluster_text_id:", sample['cluster_text_id'])

#     print("Geometry shape:", sample["geometry"].shape)
#     print("Image Features shape:", sample["image_features"].shape)
#     print("Text Features shape:", sample["text_features"].shape)
#     print("Padding Mask shape:", sample["padding_mask"].shape)
#     print("Padding Mask Image shape:", sample["padding_mask_img"].shape)
#     print("Padding Mask Text shape:", sample["padding_mask_text"].shape)
#     print("IDs shape:", len(sample["ids"]))
#     print("Type shape:", sample["cat"].shape)
#     print("Padding Mask Image shape:", sample["padding_mask_img"].shape)
#     print("cluster_image_id shape:", sample['cluster_image_id'].shape)
#     print("cluster_text_id shape:", sample['cluster_text_id'].shape)
