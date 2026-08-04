/**
 * 브라우저 렌더링 관련 순수 함수 모음.
 * DOM 조작만 담당하며, 이벤트 배선(wiring)은 app.js에서 처리한다.
 * marked/DOMPurify 인스턴스를 인자로 주입받아 테스트(Vitest+jsdom)에서도
 * 브라우저와 동일한 정화 로직을 검증할 수 있게 한다.
 */

// 링크는 http/https만 허용한다 (javascript:, data: 등 차단).
export const SANITIZE_CONFIG = {
    ALLOWED_URI_REGEXP: /^https?:\/\//i
};

/**
 * 모든 <a> 태그에 target="_blank" + rel="noopener noreferrer"를 강제한다.
 * DOMPurify 인스턴스당 한 번만 등록하면 된다.
 */
export function installLinkSafetyHook(DOMPurify) {
    DOMPurify.addHook('afterSanitizeAttributes', (node) => {
        if (node.tagName === 'A' && node.hasAttribute('href')) {
            node.setAttribute('target', '_blank');
            node.setAttribute('rel', 'noopener noreferrer');
        }
    });
}

/**
 * 신뢰할 수 없는 외부(웹 검색) 콘텐츠가 섞인 답변을 Markdown -> HTML로 변환하고 정화한다.
 */
export function sanitizeAnswerHtml(rawAnswer, { marked, DOMPurify }) {
    return DOMPurify.sanitize(marked.parse(rawAnswer), SANITIZE_CONFIG);
}

/**
 * 오류 메시지를 HTML 문자열 조합 없이 안전하게 렌더링한다.
 */
export function showMessage(container, className, title, message) {
    container.innerHTML = '';
    const div = document.createElement('div');
    div.className = className;
    const strong = document.createElement('strong');
    strong.textContent = title;
    div.appendChild(strong);
    div.appendChild(document.createTextNode(' ' + message));
    container.appendChild(div);
}

/**
 * 출처 목록을 렌더링한다. source.source/content 등은 외부(웹 검색) 값일 수 있으므로
 * 항상 textContent로만 삽입해 HTML로 해석되지 않도록 한다.
 *
 * @returns {{ hasMore: boolean }} maxInitial을 초과하는 출처가 있는지 여부
 */
export function renderSources(sourcesContent, sources, { isExpanded, maxInitial }) {
    if (!sources || sources.length === 0) {
        return { hasMore: false };
    }

    const displayCount = isExpanded ? sources.length : Math.min(maxInitial, sources.length);

    sourcesContent.innerHTML = '';
    sources.forEach((source, index) => {
        const pageInfo = source.page ? ` (페이지 ${source.page})` : '';

        const item = document.createElement('div');
        item.className = 'source-item' + (index >= displayCount ? ' hidden' : '');

        const header = document.createElement('div');
        header.className = 'source-header';
        header.textContent = `[${source.index}] ${source.source}${pageInfo}`;

        const content = document.createElement('div');
        content.className = 'source-content';
        content.textContent = `${source.content}...`;

        item.appendChild(header);
        item.appendChild(content);
        sourcesContent.appendChild(item);
    });

    return { hasMore: sources.length > maxInitial };
}
