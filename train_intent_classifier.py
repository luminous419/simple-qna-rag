#!/usr/bin/env python3
"""
Intent Classifier 학습 배치 프로그램

BAAI/bge-m3 임베딩 모델 + 분류 헤드(Softmax)를 사용하여
사용자 질문의 의도를 분류하는 모델을 학습합니다.

의도 라벨:
- explanation: 개념 설명
- comparison: 비교
- procedure: 절차 설명
- yesno: 예/아니오 질문
- other: 기타
- uncertain: 불명확

출력: intent-bge-m3-softmax/ 디렉토리에 학습된 모델 저장
"""

import json
import os
from typing import List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score
import numpy as np


# 설정
@dataclass
class TrainConfig:
    # 모델
    embedding_model: str = "BAAI/bge-m3"
    max_seq_len: int = 256

    # 학습
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # 데이터
    train_path: str = "intent_dataset/train.jsonl"
    dev_path: str = "intent_dataset/dev.jsonl"

    # 출력
    output_dir: str = "intent-bge-m3-softmax"

    # 디바이스
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 라벨
    labels: List[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = ["explanation", "comparison", "procedure", "yesno", "other", "uncertain"]


# 데이터셋 클래스
class IntentDataset(Dataset):
    def __init__(self, data: List[Dict], label_to_id: Dict[str, int]):
        self.data = data
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "text": item["text"],
            "label": self.label_to_id[item["label"]]
        }


# 분류 모델
class IntentClassifier(nn.Module):
    def __init__(self, embedding_model: SentenceTransformer, num_labels: int, embedding_dim: int = 1024):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = nn.Linear(embedding_dim, num_labels)

        # 임베딩 모델 freeze (선택적)
        # for param in self.embedding_model.parameters():
        #     param.requires_grad = False

    def forward(self, texts: List[str]):
        # 임베딩 생성
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Clone embeddings to allow gradients (sentence-transformers uses inference mode)
        embeddings = embeddings.clone().detach().requires_grad_(True)

        # 분류
        logits = self.classifier(embeddings)
        return logits


def load_dataset(file_path: str) -> List[Dict]:
    """JSONL 파일에서 데이터 로드"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def collate_fn(batch):
    """DataLoader용 collate 함수"""
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    return {"texts": texts, "labels": labels}


def train_epoch(model, dataloader, optimizer, criterion, device):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        texts = batch["texts"]
        labels = batch["labels"].to(device)

        # Forward
        logits = model(texts)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, f1


def evaluate(model, dataloader, criterion, device, id_to_label):
    """검증"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            # Forward
            logits = model(texts)
            loss = criterion(logits, labels)

            # 통계
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Classification Report
    label_names = [id_to_label[i] for i in range(len(id_to_label))]
    report = classification_report(all_labels, all_preds, target_names=label_names)

    return avg_loss, f1, report


def main():
    config = TrainConfig()

    print("=" * 60)
    print("Intent Classifier 학습 시작")
    print("=" * 60)
    print(f"모델: {config.embedding_model}")
    print(f"디바이스: {config.device}")
    print(f"배치 크기: {config.batch_size}")
    print(f"에폭: {config.epochs}")
    print(f"학습률: {config.learning_rate}")
    print("=" * 60)

    # 라벨 매핑
    label_to_id = {label: idx for idx, label in enumerate(config.labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    print(f"\n라벨: {config.labels}")
    print(f"라벨 수: {len(config.labels)}")

    # 데이터 로드
    print(f"\n데이터 로딩 중...")
    train_data = load_dataset(config.train_path)
    dev_data = load_dataset(config.dev_path)

    print(f"Train: {len(train_data)}개")
    print(f"Dev: {len(dev_data)}개")

    # 데이터셋 생성
    train_dataset = IntentDataset(train_data, label_to_id)
    dev_dataset = IntentDataset(dev_data, label_to_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 모델 초기화
    print(f"\n임베딩 모델 로딩 중: {config.embedding_model}")
    embedding_model = SentenceTransformer(config.embedding_model)
    embedding_model.max_seq_length = config.max_seq_len

    # 임베딩 차원 확인
    test_embedding = embedding_model.encode(["test"], show_progress_bar=False)
    embedding_dim = test_embedding.shape[1]
    print(f"임베딩 차원: {embedding_dim}")

    # 분류 모델 생성
    model = IntentClassifier(embedding_model, len(config.labels), embedding_dim)
    model = model.to(config.device)

    # 옵티마이저 & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    # 학습
    print(f"\n학습 시작...")
    best_f1 = 0.0

    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, config.device)
        print(f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}")

        # Evaluate
        dev_loss, dev_f1, dev_report = evaluate(model, dev_loader, criterion, config.device, id_to_label)
        print(f"Dev Loss: {dev_loss:.4f}, F1: {dev_f1:.4f}")
        print(f"\nClassification Report:\n{dev_report}")

        # Best model 저장
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            print(f"✅ Best F1 업데이트: {best_f1:.4f}")

            # 모델 저장
            os.makedirs(config.output_dir, exist_ok=True)

            # 분류기 헤드만 저장 (임베딩 모델은 HuggingFace Hub에서 로드)
            torch.save(
                model.classifier.state_dict(),
                os.path.join(config.output_dir, "classifier_head.pt")
            )

            # 설정 저장 (임베딩 모델 이름 포함)
            config_dict = {
                "labels": config.labels,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "embedding_dim": embedding_dim,
                "num_labels": len(config.labels),
                "embedding_model_name": config.embedding_model  # HuggingFace 모델 이름 저장
            }

            with open(os.path.join(config.output_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)

            print(f"모델 저장 완료: {config.output_dir}")

    print(f"\n{'='*60}")
    print(f"✅ 학습 완료!")
    print(f"Best Dev F1: {best_f1:.4f}")
    print(f"모델 저장 위치: {config.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
