// Shared test setup, run before every test file (vite.config.ts -> test.setupFiles).
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'
// Adds DOM-specific checks to Vitest's expect(), e.g. toBeDisabled().
import '@testing-library/jest-dom/vitest'

// Remove whatever the last test rendered. Testing Library only does this by itself when
// the test runner exposes global hooks; without it, every test's form piles up on the
// same page and lookups find several matches.
afterEach(() => cleanup())
