#!/usr/bin/env python3
"""
Intent Classification 학습 데이터 생성기

인터넷 IT 회사의 실제 업무 시나리오를 반영한 질문 데이터를 생성합니다.
"""

import json
import random

# ============================================================================
# 데이터 템플릿 정의
# ============================================================================

# 1. EXPLANATION (지식 습득) - 2000개
EXPLANATION_TEMPLATES = {
    # IT Support 관련 (500개)
    "it_support": [
        # 네트워크
        "VPN{이가} 뭔지 설명해{줘}",
        "방화벽{이} 무슨 역할을 하는지 알려{주세요}",
        "프록시 서버{가} 뭔지 {궁금합니다}",
        "DMZ 구역{의} 개념을 {설명해줘}",
        "포트 포워딩{이} 뭔지 알려{주세요}",
        "DHCP{가} 어떻게 작동하는지 {설명해줘}",
        "DNS{의} 역할을 {알려주세요}",
        "라우팅{이} 뭔지 {설명해줘}",
        "스위치와 허브의 차이를 {알려줘}",
        "게이트웨이{가} 뭔지 {설명해주세요}",
        # 보안
        "이중인증{이} 뭔지 {설명해줘}",
        "SSL/TLS{가} 무엇인지 알려{주세요}",
        "암호화{의} 원리를 {설명해줘}",
        "공인IP와 사설IP{의} 개념을 {알려주세요}",
        "MAC 주소{가} 뭔지 {설명해줘}",
        "NAT{가} 무엇인지 알려{주세요}",
        # 하드웨어
        "RAM과 ROM{의} 차이를 {설명해줘}",
        "SSD와 HDD{의} 차이를 {알려주세요}",
        "UPS{가} 뭔지 {설명해줘}",
        "블루스크린{이} 왜 뜨는지 {알려주세요}",
        # 소프트웨어
        "Active Directory{가} 뭔지 {설명해줘}",
        "그룹 정책{이} 무엇인지 {알려주세요}",
        "도메인 가입{이} 뭔지 {설명해줘}",
        "원격 데스크톱{의} 원리를 {알려주세요}",
    ],

    # HR 관련 (500개)
    "hr": [
        # 휴가/근태
        "연차{의} 개념을 {설명해줘}",
        "월차와 연차{의} 차이를 {알려주세요}",
        "대체휴가{가} 뭔지 {설명해줘}",
        "보상휴가{의} 의미를 {알려주세요}",
        "병가{가} 뭔지 {설명해줘}",
        "경조사 휴가{의} 범위를 {알려주세요}",
        "육아휴직{이} 뭔지 {설명해줘}",
        "출산휴가{의} 기간을 {알려주세요}",
        "가족돌봄휴가{가} 뭔지 {설명해줘}",
        "근태 관리{의} 원칙을 {알려주세요}",
        # 급여/복지
        "성과급{이} 뭔지 {설명해줘}",
        "인센티브 제도{를} {알려주세요}",
        "퇴직금{의} 계산법을 {설명해줘}",
        "4대보험{이} 뭔지 {알려주세요}",
        "건강검진 지원{의} 내용을 {설명해줘}",
        "사내 대출{의} 조건을 {알려주세요}",
        "학자금 지원{이} 뭔지 {설명해줘}",
        "경조사비 지원{의} 기준을 {알려주세요}",
        # 인사
        "수습 기간{이} 뭔지 {설명해줘}",
        "정규직 전환 절차{를} {알려주세요}",
        "승진 기준{에} 대해 {설명해줘}",
        "인사 평가{의} 방식을 {알려주세요}",
        "징계 절차{가} 어떻게 되는지 {설명해줘}",
        "퇴사 절차{를} {알려주세요}",
    ],

    # 인프라 관련 (500개)
    "infra": [
        # 클라우드
        "EC2{가} 뭔지 {설명해줘}",
        "S3 버킷{의} 개념을 {알려주세요}",
        "RDS{가} 무엇인지 {설명해줘}",
        "로드밸런서{의} 역할을 {알려주세요}",
        "Auto Scaling{이} 뭔지 {설명해줘}",
        "CloudFront{의} 개념을 {알려주세요}",
        "Lambda{가} 뭔지 {설명해줘}",
        "VPC{의} 개념을 {알려주세요}",
        "서브넷{이} 뭔지 {설명해줘}",
        "Security Group{의} 역할을 {알려주세요}",
        # 컨테이너/오케스트레이션
        "Docker{가} 뭔지 {설명해줘}",
        "컨테이너{의} 개념을 {알려주세요}",
        "Kubernetes{가} 무엇인지 {설명해줘}",
        "Pod{의} 역할을 {알려주세요}",
        "Deployment{가} 뭔지 {설명해줘}",
        "Service{의} 개념을 {알려주세요}",
        "Ingress{가} 뭔지 {설명해줘}",
        "Helm{의} 역할을 {알려주세요}",
        # 데이터베이스
        "Primary-Replica{가} 뭔지 {설명해줘}",
        "Sharding{의} 개념을 {알려주세요}",
        "Replication{이} 무엇인지 {설명해줘}",
        "인덱스{의} 역할을 {알려주세요}",
        "트랜잭션{이} 뭔지 {설명해줘}",
        "정규화{의} 개념을 {알려주세요}",
        # 모니터링
        "APM{이} 뭔지 {설명해줘}",
        "로그 수집{의} 원리를 {알려주세요}",
        "메트릭{이} 무엇인지 {설명해줘}",
        "알람 설정{의} 방법을 {알려주세요}",
    ],

    # 조직/프로세스 (500개)
    "organization": [
        "IT Support 팀{의} 역할을 {설명해줘}",
        "인프라 팀{이} 무슨 일을 하는지 {알려주세요}",
        "보안 팀{의} 업무를 {설명해줘}",
        "DBA팀{이} 무엇을 하는지 {알려주세요}",
        "개발팀과 인프라팀{의} 협업 방식을 {설명해줘}",
        "DevOps{가} 뭔지 {알려주세요}",
        "SRE{의} 역할을 {설명해줘}",
        "QA팀{이} 무슨 일을 하는지 {알려주세요}",
        "변경관리 프로세스{를} {설명해줘}",
        "릴리스 프로세스{의} 절차를 {알려주세요}",
        "장애 대응 절차{를} {설명해줘}",
        "사고 보고서{가} 뭔지 {알려주세요}",
        "코드 리뷰{의} 목적을 {설명해줘}",
        "CI/CD{가} 무엇인지 {알려주세요}",
    ],
}

# 2. COMPARISON (비교) - 2000개
COMPARISON_TEMPLATES = {
    "it_support": [
        "VPN과 원격 데스크톱{을} 비교해{줘}",
        "WiFi와 유선{의} 장단점을 {표로 보여주세요}",
        "Windows와 Mac{을} 비교해서 {설명해줘}",
        "Chrome과 Edge{의} 차이점을 {알려주세요}",
        "SSD와 HDD{를} 비교해{줘}",
        "RAM 8GB와 16GB{의} 차이를 {표로 정리해주세요}",
        "데스크톱과 노트북{을} 비교해서 {설명해줘}",
        "공유기와 스위치{의} 차이를 {알려주세요}",
        "USB-C와 USB 3.0{을} 비교해{줘}",
        "블루투스와 WiFi{의} 장단점을 {표로 보여주세요}",
    ],
    "hr": [
        "연차와 월차{를} 비교해{줘}",
        "정규직과 계약직{의} 차이를 {표로 보여주세요}",
        "육아휴직과 출산휴가{를} 비교해서 {설명해줘}",
        "재택근무와 사무실 근무{의} 장단점을 {알려주세요}",
        "경조사 휴가와 병가{를} 비교해{줘}",
        "시간외 근무와 휴일 근무{의} 차이를 {표로 정리해주세요}",
        "성과급과 인센티브{를} 비교해서 {설명해줘}",
        "반차와 시간연차{의} 차이를 {알려주세요}",
        "전일 연차와 반일 연차{를} 비교해{줘}",
        "보상휴가와 대체휴가{의} 장단점을 {표로 보여주세요}",
    ],
    "infra": [
        "AWS와 GCP{를} 비교해{줘}",
        "MySQL과 PostgreSQL{의} 장단점을 {표로 보여주세요}",
        "Redis와 Memcached{를} 비교해서 {설명해줘}",
        "Nginx와 Apache{의} 차이를 {알려주세요}",
        "Docker와 VM{을} 비교해{줘}",
        "Kubernetes와 Docker Swarm{의} 차이를 {표로 정리해주세요}",
        "REST API와 GraphQL{을} 비교해서 {설명해줘}",
        "HTTP와 HTTPS{의} 장단점을 {알려주세요}",
        "TCP와 UDP{를} 비교해{줘}",
        "IPv4와 IPv6{의} 차이를 {표로 보여주세요}",
    ],
}

# 3. PROCEDURE (절차) - 2000개
PROCEDURE_TEMPLATES = {
    "it_support": [
        "VPN 연결하는 절차를 단계별로 {알려줘}",
        "비밀번호 초기화는 어떻게 {하나요}",
        "사내 인터넷이 안될 때 해결 방법을 {알려주세요}",
        "노트북 수리 신청 절차를 {설명해줘}",
        "소프트웨어 설치 요청은 어떻게 {하나요}",
        "이메일 계정 추가 방법을 {알려주세요}",
        "프린터 연결하는 절차를 단계별로 {설명해줘}",
        "회의실 예약은 어떻게 {하나요}",
        "사내 WiFi 연결 방법을 {알려주세요}",
        "공용 폴더 권한 신청 절차를 {설명해줘}",
        "메일 서명 설정하는 방법을 {알려줘}",
        "VPN 재연결 절차를 단계별로 {알려주세요}",
        "모니터 추가 신청은 어떻게 {하나요}",
        "키보드/마우스 교체 요청 절차를 {설명해줘}",
        "노트북 초기화 방법을 {알려주세요}",
    ],
    "hr": [
        "연차 신청하는 방법을 {알려줘}",
        "경조사 휴가 신청 절차를 단계별로 {설명해줘}",
        "출장 신청은 어떻게 {하나요}",
        "급여명세서 조회 방법을 {알려주세요}",
        "증명서 발급 절차를 {설명해줘}",
        "이직 추천서 요청은 어떻게 {하나요}",
        "건강검진 신청 방법을 {알려주세요}",
        "사내 대출 신청 절차를 단계별로 {설명해줘}",
        "학자금 지원 신청은 어떻게 {하나요}",
        "경조사비 청구 방법을 {알려주세요}",
        "퇴직금 조회 절차를 {설명해줘}",
        "4대보험 가입 증명서 발급은 어떻게 {하나요}",
        "재직증명서 발급 방법을 {알려주세요}",
        "인사평가 입력 절차를 단계별로 {설명해줘}",
        "육아휴직 신청은 어떻게 {하나요}",
    ],
    "infra": [
        "방화벽 오픈 신청 절차를 단계별로 {알려줘}",
        "서버 계정 신청은 어떻게 {하나요}",
        "DB 접근 권한 요청 방법을 {알려주세요}",
        "신규 서버 론칭 절차를 {설명해줘}",
        "도메인 등록 신청은 어떻게 {하나요}",
        "SSL 인증서 발급 방법을 {알려주세요}",
        "배포 권한 신청 절차를 단계별로 {설명해줘}",
        "로그 조회 권한 요청은 어떻게 {하나요}",
        "모니터링 알람 설정 방법을 {알려주세요}",
        "백업 복구 절차를 {설명해줘}",
        "장애 대응 프로세스를 단계별로 {알려줘}",
        "긴급 배포 절차는 어떻게 {되나요}",
        "서버 증설 신청 방법을 {알려주세요}",
        "CDN 설정 절차를 {설명해줘}",
        "데이터베이스 마이그레이션 절차를 단계별로 {알려줘}",
    ],
}

# 4. YESNO (예/아니오) - 2000개
YESNO_TEMPLATES = {
    "it_support": [
        "VPN 없이 사내망 접속 {되나요}",
        "개인 노트북 사용 {가능한가요}",
        "회사 이메일을 개인 폰에서 볼 수 {있어}",
        "주말에도 IT 지원 {가능해}",
        "원격으로 프린터 사용할 수 {있나요}",
        "사내 WiFi가 5GHz를 {지원해}",
        "듀얼 모니터 설치 {가능한가요}",
        "개인 클라우드 스토리지 사용 {가능해}",
        "사무실에 무선 충전기 {있어}",
        "맥북 지급 {가능한가요}",
    ],
    "hr": [
        "사내 대출 신청할 수 {있어}",
        "재택근무 신청 {가능한가요}",
        "육아휴직 남자도 쓸 수 {있어}",
        "연차를 시간 단위로 쓸 수 {있나요}",
        "병가 사용할 때 진단서 {필요해}",
        "점심시간에 외출 {가능한가요}",
        "경조사비 당일 신청 {가능해}",
        "출산휴가는 유급{인가요}",
        "반차 사용할 때 사유 작성 {필요해}",
        "휴가 취소 {가능한가요}",
    ],
    "infra": [
        "운영 DB 직접 접속 {가능해}",
        "프로덕션 서버에 배포 권한 {있나요}",
        "개발 서버에서 외부 API 호출 {가능해}",
        "로컬에서 VPN으로 DB 접속 {되나요}",
        "서버 로그 실시간 조회 {가능한가요}",
        "긴급 배포 승인 없이 {가능해}",
        "클라우드 비용 조회 {가능한가요}",
        "테스트 서버 직접 재시작 {가능해}",
        "방화벽 규칙 수정 권한 {있나요}",
        "백업 데이터 다운로드 {가능해}",
    ],
}

# 5. OTHER (기타) - 2000개
OTHER_TEMPLATES = [
    "아까 말한 거 다시 {정리해줘}",
    "방금 전 내용을 표로 {바꿔주세요}",
    "이전 답변을 {요약해줘}",
    "코드를 JSON으로 {변환해주세요}",
    "그 내용을 불릿 포인트로 {정리해줘}",
    "위에 설명한 거 다시 {보여주세요}",
    "앞에서 말한 표를 다시 {그려줘}",
    "그거 뭐{였지}",
    "어디 팀이라고 {했지}",
    "담당자가 누구라고 {했나요}",
    "그 절차 다시 한번만 {알려줘}",
    "방금 준 링크 다시 {보여주세요}",
    "이전에 설명한 내용 {정리해줘}",
    "아까 그 항목들 다시 {나열해줘}",
    "전에 말한 조건 뭐{였어}",
    "그때 언급한 기준이 뭐{였나요}",
    "처음 말한 거 다시 한번 {설명해줘}",
    "좀 전에 준 예시 다시 {보여주세요}",
    "그 설명 간단하게 {정리해줘}",
    "위 내용을 마크다운으로 {변환해주세요}",
]

# ============================================================================
# 조사 변형 함수
# ============================================================================

def get_josa_variation():
    """다양한 조사 변형 반환"""
    return random.choice([
        {"이가": "이", "을": "을", "의": "의", "줘": "줘", "주세요": "주세요", "하나요": "하나요", "설명해줘": "설명해줘", "알려주세요": "알려주세요", "궁금합니다": "궁금합니다", "되나요": "되나요", "가능한가요": "가능한가요", "있어": "있어", "가능해": "가능해", "필요해": "필요해", "인가요": "인가요", "정리해줘": "정리해줘", "바꿔주세요": "바꿔주세요", "요약해줘": "요약해줘", "변환해주세요": "변환해주세요", "보여주세요": "보여주세요", "그려줘": "그려줘", "였지": "였지", "했지": "했지", "했나요": "했나요", "알려줘": "알려줘", "였어": "였어", "였나요": "였나요"},
        {"이가": "가", "을": "를", "의": "의", "줘": "주세요", "주세요": "줘", "하나요": "하는 건가요", "설명해줘": "설명해주세요", "알려주세요": "알려줘", "궁금합니다": "알고 싶어요", "되나요": "가능한가요", "가능한가요": "되나요", "있어": "있나요", "가능해": "가능한가요", "필요해": "필요한가요", "인가요": "인지", "정리해줘": "정리해주세요", "바꿔주세요": "바꿔줘", "요약해줘": "요약해주세요", "변환해주세요": "변환해줘", "보여주세요": "보여줘", "그려줘": "그려주세요", "였지": "였는지", "했지": "했는지", "했나요": "했어요", "알려줘": "알려주실 수 있나요", "였어": "였나", "였나요": "였죠"},
    ])

def apply_variations(template):
    """템플릿에 조사 변형 적용"""
    variations = get_josa_variation()
    result = template
    for key, value in variations.items():
        result = result.replace("{" + key + "}", value)
    # 남은 플레이스홀더 제거
    import re
    result = re.sub(r'\{[^}]+\}', '', result)
    return result

# ============================================================================
# 데이터 생성 함수
# ============================================================================

def generate_explanation_data(count=2000):
    """Explanation 카테고리 데이터 생성"""
    data = []
    templates = []
    
    # 모든 템플릿 수집
    for category, temp_list in EXPLANATION_TEMPLATES.items():
        templates.extend(temp_list)
    
    # count개 생성 (중복 허용하여 다양성 확보)
    for _ in range(count):
        template = random.choice(templates)
        text = apply_variations(template)
        data.append({"text": text, "label": "explanation"})
    
    return data

def generate_comparison_data(count=2000):
    """Comparison 카테고리 데이터 생성"""
    data = []
    templates = []
    
    for category, temp_list in COMPARISON_TEMPLATES.items():
        templates.extend(temp_list)
    
    for _ in range(count):
        template = random.choice(templates)
        text = apply_variations(template)
        data.append({"text": text, "label": "comparison"})
    
    return data

def generate_procedure_data(count=2000):
    """Procedure 카테고리 데이터 생성"""
    data = []
    templates = []
    
    for category, temp_list in PROCEDURE_TEMPLATES.items():
        templates.extend(temp_list)
    
    for _ in range(count):
        template = random.choice(templates)
        text = apply_variations(template)
        data.append({"text": text, "label": "procedure"})
    
    return data

def generate_yesno_data(count=2000):
    """YesNo 카테고리 데이터 생성"""
    data = []
    templates = []
    
    for category, temp_list in YESNO_TEMPLATES.items():
        templates.extend(temp_list)
    
    for _ in range(count):
        template = random.choice(templates)
        text = apply_variations(template)
        data.append({"text": text, "label": "yesno"})
    
    return data

def generate_other_data(count=2000):
    """Other 카테고리 데이터 생성"""
    data = []
    
    for _ in range(count):
        template = random.choice(OTHER_TEMPLATES)
        text = apply_variations(template)
        data.append({"text": text, "label": "other"})
    
    return data

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("=" * 60)
    print("Intent Classification 학습 데이터 생성 시작")
    print("=" * 60)
    
    # 각 카테고리별 2000개씩 생성
    print("\n1. Explanation 데이터 생성 중...")
    explanation_data = generate_explanation_data(2000)
    print(f"   ✅ {len(explanation_data)}개 생성 완료")
    
    print("\n2. Comparison 데이터 생성 중...")
    comparison_data = generate_comparison_data(2000)
    print(f"   ✅ {len(comparison_data)}개 생성 완료")
    
    print("\n3. Procedure 데이터 생성 중...")
    procedure_data = generate_procedure_data(2000)
    print(f"   ✅ {len(procedure_data)}개 생성 완료")
    
    print("\n4. YesNo 데이터 생성 중...")
    yesno_data = generate_yesno_data(2000)
    print(f"   ✅ {len(yesno_data)}개 생성 완료")
    
    print("\n5. Other 데이터 생성 중...")
    other_data = generate_other_data(2000)
    print(f"   ✅ {len(other_data)}개 생성 완료")
    
    # 전체 데이터 합치기
    all_data = explanation_data + comparison_data + procedure_data + yesno_data + other_data
    
    # 셔플
    random.shuffle(all_data)
    
    print(f"\n총 {len(all_data)}개 데이터 생성 완료")
    
    # Train/Dev 분리 (9:1)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    dev_data = all_data[split_idx:]
    
    print(f"Train: {len(train_data)}개")
    print(f"Dev: {len(dev_data)}개")
    
    # 파일 저장
    with open("intent_dataset/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open("intent_dataset/dev.jsonl", "w", encoding="utf-8") as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 파일 저장 완료:")
    print(f"   - intent_dataset/train.jsonl")
    print(f"   - intent_dataset/dev.jsonl")
    
    # 라벨별 분포 출력
    print("\n📊 라벨별 분포 (Train):")
    from collections import Counter
    label_counts = Counter([item["label"] for item in train_data])
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}개")
    
    print("\n" + "=" * 60)
    print("✅ 데이터 생성 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
