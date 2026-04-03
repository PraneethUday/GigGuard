import { NextResponse } from "next/server";
import { createServerClient, PLATFORM_TABLE } from "@/lib/supabase";

async function autoVerifyWorker(
  db: ReturnType<typeof createServerClient>,
  deliveryId: string,
  platforms: string[],
) {
  const matchedPlatforms: string[] = [];

  for (const platform of platforms || []) {
    const table = PLATFORM_TABLE[platform];
    if (!table) continue;

    const { data } = await db
      .from(table)
      .select("worker_id")
      .eq("worker_id", deliveryId)
      .eq("verified", true)
      .limit(1);

    if (data && data.length > 0) {
      matchedPlatforms.push(platform);
    }
  }

  return {
    auto_verified: matchedPlatforms.length > 0,
    auto_verified_platforms: matchedPlatforms,
  };
}

export async function GET() {
  try {
    const db = createServerClient();

    const { data, error } = await db
      .from("registered_workers")
      .select(
        "id, name, email, phone, city, area, platforms, tier, verification_status, delivery_id, autopay, is_active, created_at",
      )
      .order("created_at", { ascending: false })
      .limit(100);

    if (error) {
      console.error("Admin fetch error:", error);
      return NextResponse.json(
        { error: "Failed to fetch workers." },
        { status: 500 },
      );
    }

    const workersWithChecks = await Promise.all(
      (data || []).map(async (worker) => {
        const verification = await autoVerifyWorker(
          db,
          worker.delivery_id,
          worker.platforms || [],
        );
        return {
          ...worker,
          ...verification,
        };
      }),
    );

    return NextResponse.json({ workers: workersWithChecks });
  } catch (err) {
    console.error("Admin workers error:", err);
    return NextResponse.json(
      { error: "Internal server error." },
      { status: 500 },
    );
  }
}
