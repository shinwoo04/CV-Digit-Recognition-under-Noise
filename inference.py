import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import os
import sys

# 1. GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 테스트용 커스텀 데이터셋 클래스 (DaconDataset 재사용)
class DaconDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 테스트 데이터는 2번 인덱스부터 픽셀 데이터임
        image = self.df.iloc[idx, 2:].values.reshape(28, 28).astype(np.uint8)
        # ResNet 입력을 위해 3채널로 복사
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        
        if self.transform:
            image = self.transform(image)
        return image

# 3. 테스트용 이미지 변환 (학습 때와 동일한 Resize 및 Normalize 적용)
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 4. 모델 로드 및 설정 (v2 ResNet50 기준)
def load_model(model_path):
    # ResNet18 구조 생성
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    # 저장된 가중치 불러오기
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def run_inference():
    # 데이터 로드
    try:
        test_df = pd.read_csv('data/test.csv')
        submission = pd.read_csv('data/submission.csv')
    except FileNotFoundError:
        print("\n[에러] data 폴더에 csv 파일이 없습니다. 경로와 파일명을 확인해 주세요.")
        sys.exit() # 프로그램 종료
    
    # 모델 경로 지정 (v2 ResNet50 모델 사용)
    model_path = './models/best_resnet18_model.pth'
    
    if not os.path.exists(model_path):
        print(f"에러: {model_path} 파일을 찾을 수 없습니다.")
        return

    model = load_model(model_path)
    
    # DataLoader 생성
    test_dataset = DaconDataset(test_df, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 추론 시작
    preds_list = []
    print(f"추론 시작 ({device})...")
    
    with torch.inference_mode():
        for images in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds_list.extend(preds.cpu().numpy())

    # 결과 저장
    submission['digit'] = preds_list
    submission.to_csv('submission_v1_final.csv', index=False)
    print("\n추론 완료! 'submission_v1_final.csv' 파일이 저장되었습니다.")

if __name__ == "__main__":
    run_inference()