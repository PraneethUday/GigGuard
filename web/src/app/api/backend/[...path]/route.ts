import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.BACKEND_URL;

async function proxyRequest(
    req: NextRequest,
    params: Promise<{ path: string[] }>,
) {
    const { path } = await params;
    const apiPath = path.join("/");
    const qs = req.nextUrl.searchParams.toString();
    const url = `${BACKEND_URL}/api/${apiPath}${qs ? `?${qs}` : ""}`;

    try {
        const fetchOptions: RequestInit = {
            method: req.method,
            headers: { "Content-Type": "application/json" },
        };

        if (req.method !== "GET" && req.method !== "HEAD") {
            fetchOptions.body = await req.text();
        }

        const res = await fetch(url, fetchOptions);
        const contentType = res.headers.get("content-type") || "application/json";

        if (contentType.includes("application/json")) {
            const data = await res.json();
            return NextResponse.json(data, { status: res.status });
        }

        const text = await res.text();
        return new NextResponse(text, {
            status: res.status,
            headers: { "content-type": contentType },
        });
    } catch (err) {
        console.error(`Backend proxy error (${url}):`, err);
        return NextResponse.json(
            {
                error:
                    "Backend unavailable. Make sure the FastAPI server is running on " +
                    BACKEND_URL,
            },
            { status: 502 },
        );
    }
}

export async function GET(
    req: NextRequest,
    { params }: { params: Promise<{ path: string[] }> },
) {
    return proxyRequest(req, params);
}

export async function POST(
    req: NextRequest,
    { params }: { params: Promise<{ path: string[] }> },
) {
    return proxyRequest(req, params);
}

export async function PUT(
    req: NextRequest,
    { params }: { params: Promise<{ path: string[] }> },
) {
    return proxyRequest(req, params);
}

export async function PATCH(
    req: NextRequest,
    { params }: { params: Promise<{ path: string[] }> },
) {
    return proxyRequest(req, params);
}

export async function DELETE(
    req: NextRequest,
    { params }: { params: Promise<{ path: string[] }> },
) {
    return proxyRequest(req, params);
}