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
    const { action } = await req.json();

    if (!workerId || !action || !["approve", "reject"].includes(action)) {
      return NextResponse.json({ error: "Invalid request." }, { status: 400 });
    }

    const db = createServerClient();

    const { data: worker, error: fetchError } = await db
      .from("registered_workers")
      .select("id, delivery_id, platforms, verification_status")
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

    if (action === "approve" && !verification.auto_verified) {
      return NextResponse.json(
        {
          error:
            "Cannot approve: worker not found in selected platform records.",
        },
        { status: 400 },
      );
    }

    const nextStatus = action === "approve" ? "verified" : "rejected";

    const { data: updated, error: updateError } = await db
      .from("registered_workers")
      .update({ verification_status: nextStatus })
      .eq("id", workerId)
      .select("id, verification_status")
      .single();

    if (updateError) {
      return NextResponse.json(
        { error: "Failed to update worker status." },
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
