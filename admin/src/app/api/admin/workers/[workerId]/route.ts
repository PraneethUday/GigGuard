import { NextRequest, NextResponse } from "next/server";
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

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ workerId: string }> },
) {
  try {
    const { workerId } = await params;
    const body = await req.json();
    const { action, tier } = body;

    if (!workerId) {
      return NextResponse.json({ error: "Invalid request." }, { status: 400 });
    }

    const db = createServerClient();

    const { data: worker, error: fetchError } = await db
      .from("registered_workers")
      .select("id, delivery_id, platforms, verification_status, tier")
      .eq("id", workerId)
      .limit(1)
      .maybeSingle();

    if (fetchError || !worker) {
      return NextResponse.json({ error: "Worker not found." }, { status: 404 });
    }

    const verification = await autoVerifyWorker(
      db,
      worker.delivery_id,
      worker.platforms || [],
    );

    const updates: Record<string, string> = {};

    // Handle status change - admin can approve or reject regardless of auto_verified
    if (action === "approve") {
      updates.verification_status = "verified";
    } else if (action === "reject") {
      updates.verification_status = "rejected";
    } else if (action === "pending") {
      updates.verification_status = "pending";
    }

    // Handle tier change
    if (tier && ["basic", "standard", "pro"].includes(tier)) {
      updates.tier = tier;
    }

    if (Object.keys(updates).length === 0) {
      return NextResponse.json(
        { error: "No valid action or tier change provided." },
        { status: 400 },
      );
    }

    const { data: updated, error: updateError } = await db
      .from("registered_workers")
      .update(updates)
      .eq("id", workerId)
      .select("id, verification_status, tier")
      .single();

    if (updateError) {
      return NextResponse.json(
        { error: "Failed to update worker." },
        { status: 500 },
      );
    }

    return NextResponse.json({
      worker: {
        ...updated,
        ...verification,
      },
    });
  } catch (err) {
    console.error("Admin approval error:", err);
    return NextResponse.json(
      { error: "Internal server error." },
      { status: 500 },
    );
  }
}
