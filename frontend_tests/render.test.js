/**
 * XSS 방어 회귀 테스트 (Problems.md 권장 후속 작업).
 *
 * 웹 검색 결과처럼 신뢰할 수 없는 외부 콘텐츠가 answer/sources에 섞여 들어와도
 * 실행 가능한 DOM(스크립트, 이벤트 핸들러, javascript:/data: 링크)으로 렌더링되지
 * 않는지, 정상적인 https 링크에는 안전 속성이 붙는지를 검증한다.
 */
import { describe, expect, it } from 'vitest';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

import {
    installLinkSafetyHook,
    renderSources,
    sanitizeAnswerHtml,
    showMessage
} from '../static/render.js';

installLinkSafetyHook(DOMPurify);

describe('sanitizeAnswerHtml', () => {
    it('<script> 태그를 실행 가능한 형태로 남기지 않는다', () => {
        const html = sanitizeAnswerHtml('악성 답변 <script>alert(1)</script> 본문', {
            marked,
            DOMPurify
        });

        expect(html).not.toContain('<script');
        expect(html).not.toContain('alert(1)');
    });

    it('onerror 이벤트 핸들러가 포함된 img 태그를 제거한다', () => {
        const html = sanitizeAnswerHtml('<img src=x onerror=alert(1)>', { marked, DOMPurify });

        expect(html).not.toContain('onerror');
        expect(html).not.toContain('alert(1)');
    });

    it('javascript: 링크의 href를 제거한다', () => {
        const html = sanitizeAnswerHtml('[악성 링크](javascript:alert(1))', { marked, DOMPurify });

        expect(html.toLowerCase()).not.toContain('javascript:');
    });

    it('data: 링크의 href를 제거한다', () => {
        const html = sanitizeAnswerHtml(
            '[악성 링크](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)',
            { marked, DOMPurify }
        );

        expect(html.toLowerCase()).not.toContain('data:text/html');
    });

    it('정상적인 https 링크는 유지하고 noopener/noreferrer + 새 탭을 적용한다', () => {
        const html = sanitizeAnswerHtml('[안전한 링크](https://example.com/page)', {
            marked,
            DOMPurify
        });

        expect(html).toContain('href="https://example.com/page"');
        expect(html).toContain('target="_blank"');
        expect(html).toContain('rel="noopener noreferrer"');
    });

    it('이벤트 핸들러가 포함된 검색 결과 제목/요약도 무해화된다', () => {
        const maliciousTitle = '<a href="https://example.com" onmouseover="alert(1)">클릭</a>';
        const html = sanitizeAnswerHtml(`**${maliciousTitle}**\n\n요약입니다`, { marked, DOMPurify });

        expect(html).not.toContain('onmouseover');
        expect(html).not.toContain('alert(1)');
    });
});

describe('renderSources', () => {
    it('악성 source 필드를 텍스트로만 삽입하고 DOM 요소로 만들지 않는다', () => {
        const container = document.createElement('div');
        const sources = [
            {
                index: 1,
                source: '<img src=x onerror=alert(1)>',
                content: '<script>alert(2)</script>',
                page: null
            }
        ];

        renderSources(container, sources, { isExpanded: true, maxInitial: 3 });

        expect(container.querySelector('img')).toBeNull();
        expect(container.querySelector('script')).toBeNull();
        expect(container.textContent).toContain('<img src=x onerror=alert(1)>');
        expect(container.textContent).toContain('<script>alert(2)</script>');
    });

    it('maxInitial을 넘는 항목은 hidden 클래스가 붙고 hasMore가 true다', () => {
        const container = document.createElement('div');
        const sources = [1, 2, 3, 4].map((i) => ({
            index: i,
            source: `source-${i}`,
            content: `content-${i}`,
            page: null
        }));

        const { hasMore } = renderSources(container, sources, { isExpanded: false, maxInitial: 3 });
        const items = container.querySelectorAll('.source-item');

        expect(hasMore).toBe(true);
        expect(items).toHaveLength(4);
        expect(items[2].classList.contains('hidden')).toBe(false);
        expect(items[3].classList.contains('hidden')).toBe(true);
    });
});

describe('showMessage', () => {
    it('오류 메시지에 포함된 HTML을 실행하지 않고 텍스트로만 표시한다', () => {
        const container = document.createElement('div');

        showMessage(container, 'error-message', '오류 발생:', '<img src=x onerror=alert(1)>');

        expect(container.querySelector('img')).toBeNull();
        expect(container.textContent).toContain('<img src=x onerror=alert(1)>');
    });
});
