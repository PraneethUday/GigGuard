import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

async function proxyRequest(req: NextRequest, params: Promise<{ path: string[] }>) {
  const { path } = await params;
  const apiPath = path.join("/");
  const url = `${BACKEND_URL}/api/${apiPath}`;

  try {
    const fetchOptions: RequestInit = {
      method: req.method,
      headers: { "Content-Type": "application/json" },
    };

    if (req.method !== "GET" && req.method !== "HEAD") {
      fetchOptions.body = await req.text();
    }

    const res = await fetch(url, fetchOptions);
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    console.error(`Backend proxy error (${url}):`, err);
    return NextResponse.json(
      { error: "Backend unavailable. Make sure the FastAPI server is running on " + BACKEND_URL },
      { status: 502 },
    );
  }
}

export async function GET(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(req, params);
}

export async function POST(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  return proxyRequest(req, params);
}
