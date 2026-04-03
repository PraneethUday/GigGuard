import { NextRequest, NextResponse } from "next/server";
import { createServerClient, PLATFORM_TABLE } from "@/lib/supabase";

export async function POST(req: NextRequest) {
  try {
    const { deliveryId, platforms } = await req.json();

    if (!deliveryId || !platforms?.length) {
      return NextResponse.json(
        { error: "Delivery ID and at least one platform are required." },
        { status: 400 },
      );
    }

    const db = createServerClient();
    const matchedPlatforms: string[] = [];

    for (const platform of platforms) {
      const table = PLATFORM_TABLE[platform];
      if (!table) continue;

      const { data } = await db
        .from(table)
        .select("worker_id")
        .eq("worker_id", deliveryId)
        .limit(1);

      if (data && data.length > 0) {
        matchedPlatforms.push(platform);
      }
    }

    return NextResponse.json({
      verified: matchedPlatforms.length > 0,
      matched_platforms: matchedPlatforms,
      checked_platforms: platforms,
    });
  } catch (err) {
    console.error("Verify ID error:", err);
    return NextResponse.json(
      { error: "Verification failed." },
      { status: 500 },
    );
  }
}
