#!/usr/bin/env node
/**
 * package-lock.json에 고정된 marked/DOMPurify 버전을 web/static/vendor/로 복사한다.
 * 프런트엔드 테스트(tests/frontend/)가 import하는 npm 패키지와 브라우저가
 * 실제로 로드하는 파일이 항상 동일한 바이트가 되도록 보장하기 위함이다
 * (CDN 참조 시 발생하던 버전 불일치 문제 해결).
 *
 * npm install 이후 자동 실행되며, `npm run sync-vendor`로 수동 실행도 가능하다.
 */
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const vendorDir = join(root, 'web', 'static', 'vendor');

const files = [
    'node_modules/marked/lib/marked.umd.js',
    'node_modules/marked/lib/marked.umd.js.map',
    'node_modules/dompurify/dist/purify.min.js',
    'node_modules/dompurify/dist/purify.min.js.map'
];

mkdirSync(vendorDir, { recursive: true });

for (const relativeSrc of files) {
    const destName = relativeSrc.split('/').pop();
    copyFileSync(join(root, relativeSrc), join(vendorDir, destName));
}

console.log(`web/static/vendor/에 ${files.length}개 파일을 동기화했습니다.`);
