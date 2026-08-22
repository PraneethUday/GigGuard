import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ deliveryId: string }> },
) {
  const { deliveryId } = await params;
  const backendUrl = `${BACKEND_URL}/api/claims/worker/${encodeURIComponent(deliveryId)}`;

  try {
    const res = await fetch(backendUrl);
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { error: "Backend unavailable.", data: [] },
      { status: 502 },
    );
  }
}
