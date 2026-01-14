import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# 1. 데이터 불러오기
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 2. 데이터 기본 정보 출력
print("--- Train Data Information ---")
print(train.head())
print(f"Train Shape: {train.shape}")

print("\n--- Test Data Information ---")
print(test.head())
print(f"Test Shape: {test.shape}")   # (20480, 786)

# 3. 데이터 시각화 (첫 번째 숫자 확인해보기)
# 0~783번 컬럼이 픽셀 데이터이므로, 이를 28x28 크기로 바꿉니다.
idx = 0
img = train.iloc[idx, 3:].values.reshape(28, 28).astype(float)
digit = train.iloc[idx, 1]  # 숫자 정답
letter = train.iloc[idx, 2] # 배경에 깔린 글자

'''
plt.imshow(img, cmap='gray')
plt.title(f"Index: {idx}, Digit: {digit}, Letter: {letter}")
plt.show()
'''

# 4. 학습/검증 데이터 분리 (8:2)
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42, stratify=train['digit'])

# 5. 커스텀 데이터셋 클래스 정의
class DaconDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.is_test:
            # 테스트 데이터는 2번 인덱스부터 픽셀
            image = self.df.iloc[idx, 2:].values.reshape(28, 28).astype(np.uint8)
            label = -1 # 테스트는 정답이 없으므로 더미값
        else:
            # 트레인 데이터는 3번 인덱스부터 픽셀
            image = self.df.iloc[idx, 3:].values.reshape(28, 28).astype(np.uint8)
            label = self.df.iloc[idx, 1] # digit
        
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
# 6. 이미지 변환(Transform) 설정
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 7. DataLoader 생성
train_dataset = DaconDataset(train_df, transform=data_transforms)
val_dataset = DaconDataset(val_df, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. 전이학습 모델 불러오기 
model = models.resnet18(pretrained=True)

# 마지막 출력층을 우리 문제(0~9, 10개 클래스)에 맞게 수정
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 6. 손실함수(Loss) 및 최적화(Optimizer) 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 20
best_val_acc = 0.0

print(f"\n{device} 장치에서 학습을 시작합니다...")

# 7. 모델 학습 및 검증
for epoch in tqdm(range(EPOCHS)):
   
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        
        outputs = model(images) 
        loss = criterion(outputs, labels) 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
    
    epoch_train_loss = train_loss / len(train_loader.dataset)
    
    model.eval()
    val_corrects = 0
    with torch.inference_mode(): 
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            
    epoch_val_acc = val_corrects.double() / len(val_loader.dataset)
    
    # 결과 출력
    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Acc: {epoch_val_acc:.4f}')
    
    # 베스트 모델 저장 (Best Model Saving)
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        save_path = './models'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'best_resnet18_model.pth'))
        print(f"==> Best Model Saved in '{save_path}'! (Acc: {best_val_acc:.4f})")

print(f"\n학습 완료! 최고 검증 정확도: {best_val_acc:.4f}")
