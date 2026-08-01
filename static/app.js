/**
 * 페이지 부트스트랩: DOM 엘리먼트 참조, 이벤트 배선, /rag 호출.
 * 정화/렌더링 로직 자체는 render.js에 있고, 여기서는 그것들을 실제 DOM/전역
 * marked·DOMPurify(static/vendor/, scripts/sync-vendor.js로 동기화됨)와 연결한다.
 */
import { installLinkSafetyHook, sanitizeAnswerHtml, showMessage, renderSources } from './render.js';

const questionInput = document.getElementById('questionInput');
const searchButton = document.getElementById('searchButton');
const loading = document.getElementById('loading');
const resultContainer = document.getElementById('resultContainer');
const answerContent = document.getElementById('answerContent');
const sourcesSection = document.getElementById('sourcesSection');
const sourcesContent = document.getElementById('sourcesContent');
const showMoreContainer = document.getElementById('showMoreContainer');
const showMoreBtn = document.getElementById('showMoreBtn');

let allSources = [];
let isExpanded = false;
const MAX_INITIAL_SOURCES = 3;

// marked/DOMPurify 로드 실패(정적 자산 배포 문제 등) 시 조용히 깨지는 대신
// 명시적으로 알리고 검색 기능을 비활성화한다.
if (!window.marked || !window.DOMPurify) {
    console.error('marked 또는 DOMPurify를 불러오지 못했습니다.');
    showMessage(
        answerContent,
        'error-message',
        '페이지 로드 오류:',
        '필수 스크립트를 불러오지 못했습니다. 새로고침해도 계속되면 관리자에게 문의하세요.'
    );
    resultContainer.classList.add('active');
    searchButton.disabled = true;
    questionInput.disabled = true;
} else {
    installLinkSafetyHook(window.DOMPurify);

    // Enter 키 이벤트
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });

    // 검색 버튼 클릭
    searchButton.addEventListener('click', handleSearch);

    // 더보기 버튼 클릭
    showMoreBtn.addEventListener('click', toggleSources);
}

function toggleSources() {
    isExpanded = !isExpanded;
    renderCurrentSources();

    // 버튼 텍스트 및 아이콘 변경
    const icon = showMoreBtn.querySelector('.icon');
    const text = showMoreBtn.querySelector('.text');

    if (isExpanded) {
        icon.textContent = '−';
        text.textContent = '접기';
        showMoreBtn.classList.add('expanded');
    } else {
        icon.textContent = '+';
        text.textContent = '더보기';
        showMoreBtn.classList.remove('expanded');
    }
}

function renderCurrentSources() {
    if (!allSources || allSources.length === 0) {
        sourcesSection.style.display = 'none';
        return;
    }

    const { hasMore } = renderSources(sourcesContent, allSources, {
        isExpanded,
        maxInitial: MAX_INITIAL_SOURCES
    });
    showMoreContainer.classList.toggle('hidden', !hasMore);
    sourcesSection.style.display = 'block';
}

async function handleSearch() {
    const question = questionInput.value.trim();

    if (!question) {
        alert('질문을 입력해주세요.');
        return;
    }

    // UI 상태 변경
    searchButton.disabled = true;
    loading.classList.add('active');
    resultContainer.classList.remove('active');

    try {
        // API 호출
        const response = await fetch('/rag', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        if (data.success) {
            // 답변 표시 (Markdown 렌더링 후 신뢰할 수 없는 콘텐츠 정화)
            answerContent.innerHTML = sanitizeAnswerHtml(data.answer, {
                marked: window.marked,
                DOMPurify: window.DOMPurify
            });

            // 출처 표시 (초기 상태로 리셋)
            isExpanded = false;
            allSources = data.sources || [];
            if (allSources.length > 0) {
                renderCurrentSources();
            } else {
                sourcesSection.style.display = 'none';
                showMoreContainer.classList.add('hidden');
            }

            resultContainer.classList.add('active');
        } else {
            // 오류 표시
            showMessage(answerContent, 'error-message', '오류 발생:', data.answer);
            sourcesSection.style.display = 'none';
            resultContainer.classList.add('active');
        }

    } catch (error) {
        console.error('Error:', error);
        showMessage(answerContent, 'error-message', '서버 오류:', error.message);
        sourcesSection.style.display = 'none';
        resultContainer.classList.add('active');
    } finally {
        // UI 상태 복원
        searchButton.disabled = false;
        loading.classList.remove('active');
    }
}
