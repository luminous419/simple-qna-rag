#!/usr/bin/env python3
"""
Intent Classifier 추론 모듈

학습된 Intent Classifier 모델을 로드하여 사용자 질문의 의도를 분류합니다.
"""

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from simple_qna_rag.config import INTENT_MODEL_PATH


# 전역 캐시 (모듈 레벨 싱글톤)
_classifier_cache = {
    "model": None,
    "config": None,
    "embedding_model": None
}


class IntentClassifier(nn.Module):
    """Intent 분류 모델"""
    def __init__(self, embedding_model: SentenceTransformer, num_labels: int, embedding_dim: int = 1024):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = nn.Linear(embedding_dim, num_labels)

    def forward(self, texts):
        # 임베딩 생성
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # 디바이스 일치시키기
        embeddings = embeddings.to(self.classifier.weight.device)

        # 분류
        logits = self.classifier(embeddings)
        return logits


def load_intent_classifier(model_dir: str = INTENT_MODEL_PATH):
    """
    학습된 Intent Classifier 모델 로드 (캐싱 적용)

    Args:
        model_dir: 학습된 모델이 저장된 디렉토리

    Returns:
        tuple: (model, config)
    """
    # 이미 로드된 경우 캐시 반환
    if _classifier_cache["model"] is not None:
        return _classifier_cache["model"], _classifier_cache["config"]

    # 모델 디렉토리 확인
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Intent Classifier 모델을 찾을 수 없습니다: {model_dir}\n"
            f"먼저 train_intent_classifier.py를 실행하여 모델을 학습하세요."
        )

    # 설정 파일 로드
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # M4.1 REPLACE(Design.md §6.1) — 모델/임베딩 절대경로는 콘솔/로그에
    # 출력하지 않는다(REQ-003.3).
    # 임베딩 모델 이름 확인
    embedding_model_name = config.get("embedding_model_name")
    if not embedding_model_name:
        # 이전 버전 호환성: embedding_model 폴더가 있으면 그것을 사용
        embedding_model_path = os.path.join(model_dir, "embedding_model")
        if os.path.exists(embedding_model_path):
            embedding_model = SentenceTransformer(embedding_model_path)
        else:
            raise FileNotFoundError(
                f"설정 파일에 embedding_model_name이 없고, "
                f"로컬 임베딩 모델도 찾을 수 없습니다: {embedding_model_path}"
            )
    else:
        # HuggingFace Hub에서 직접 로드
        embedding_model = SentenceTransformer(embedding_model_name)

    # 분류 모델 생성
    model = IntentClassifier(
        embedding_model=embedding_model,
        num_labels=config["num_labels"],
        embedding_dim=config["embedding_dim"]
    )

    # 분류기 헤드 가중치 로드
    classifier_head_path = os.path.join(model_dir, "classifier_head.pt")
    if not os.path.exists(classifier_head_path):
        raise FileNotFoundError(f"분류기 헤드를 찾을 수 없습니다: {classifier_head_path}")

    model.classifier.load_state_dict(torch.load(classifier_head_path, map_location='cpu', weights_only=True))

    # CPU로 이동 및 평가 모드로 전환
    model = model.to('cpu')
    model.eval()

    # 캐시에 저장
    _classifier_cache["model"] = model
    _classifier_cache["config"] = config
    _classifier_cache["embedding_model"] = embedding_model

    return model, config


def classify_intent(question: str, model_dir: str = INTENT_MODEL_PATH) -> str:
    """
    사용자 질문의 의도를 분류

    Args:
        question: 사용자 질문 텍스트
        model_dir: 학습된 모델 디렉토리

    Returns:
        str: 예측된 intent 라벨 (explanation/comparison/procedure/yesno/other/uncertain)
    """
    # 모델 로드 (캐싱됨)
    model, config = load_intent_classifier(model_dir)

    # 추론
    with torch.no_grad():
        logits = model([question])
        pred_id = torch.argmax(logits, dim=1).item()

    # ID를 라벨로 변환
    predicted_label = config["id_to_label"][str(pred_id)]

    return predicted_label


def classify_intent_with_confidence(question: str, model_dir: str = INTENT_MODEL_PATH) -> tuple[str, float]:
    """
    사용자 질문의 의도를 분류하고 신뢰도 반환

    Args:
        question: 사용자 질문 텍스트
        model_dir: 학습된 모델 디렉토리

    Returns:
        tuple: (predicted_label, confidence_score)
    """
    # 모델 로드 (캐싱됨)
    model, config = load_intent_classifier(model_dir)

    # 추론
    with torch.no_grad():
        logits = model([question])
        probs = torch.softmax(logits, dim=1)
        confidence, pred_id = torch.max(probs, dim=1)
        pred_id = pred_id.item()
        confidence = confidence.item()

    # ID를 라벨로 변환
    predicted_label = config["id_to_label"][str(pred_id)]

    return predicted_label, confidence
