# GigGuard Admin Server

Standalone Next.js server for insurer admin operations.

## Run locally

1. Install dependencies:

```bash
npm install
```

2. Configure environment:

```bash
cp .env.example .env.local
```

Fill these values in `.env.local`:

- `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

3. Start admin server:

```bash
npm run dev
```

Admin dashboard runs at `http://localhost:3001/admin`.

## Build

```bash
npm run build
npm run start
```
