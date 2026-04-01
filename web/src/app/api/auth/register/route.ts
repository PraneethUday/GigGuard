import { NextRequest, NextResponse } from "next/server";
import { createServerClient, PLATFORM_TABLE } from "@/lib/supabase";
import { hashPassword, createToken } from "@/lib/auth";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      name,
      age,
      phone,
      email,
      password,
      city,
      area,
      deliveryId,
      platforms,
      pan,
      aadhaar,
      upi,
      bank,
      consent,
      gpsConsent,
      autopay,
      tier,
    } = body;

    // Basic validation
    if (
      !name ||
      !email ||
      !phone ||
      !password ||
      !platforms?.length ||
      !deliveryId
    ) {
      return NextResponse.json(
        { error: "Missing required fields." },
        { status: 400 },
      );
    }
    if (password.length < 6) {
      return NextResponse.json(
        { error: "Password must be at least 6 characters." },
        { status: 400 },
      );
    }

    const db = createServerClient();

    // Check for duplicate email or phone
    const { data: existing } = await db
      .from("registered_workers")
      .select("id")
      .or(`email.eq.${email},phone.eq.${phone}`)
      .limit(1);

    if (existing && existing.length > 0) {
      return NextResponse.json(
        { error: "An account with this email or phone already exists." },
        { status: 409 },
      );
    }

    // Auto-check deliveryId against selected platform tables.
    // Final approval is done by admin, so status remains pending here.
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
    const autoVerified = matchedPlatforms.length > 0;

    // Hash password and create user
    const passwordHash = await hashPassword(password);

    const { data: newUser, error: insertError } = await db
      .from("registered_workers")
      .insert({
        name,
        age: Number(age),
        phone,
        email,
        password_hash: passwordHash,
        city,
        area,
        delivery_id: deliveryId,
        platforms,
        pan: pan || null,
        aadhaar: aadhaar || null,
        upi: upi || null,
        bank: bank || null,
        consent: !!consent,
        gps_consent: !!gpsConsent,
        autopay: !!autopay,
        tier: tier || "standard",
        verification_status: "pending",
      })
      .select("id, name, email, platforms, tier, verification_status, city")
      .single();

    if (insertError) {
      console.error("Insert error:", insertError);
      return NextResponse.json(
        { error: "Registration failed. Please try again." },
        { status: 500 },
      );
    }

    const token = await createToken(newUser.id);

    return NextResponse.json({
      token,
      user: newUser,
      verified: false,
      auto_verified: autoVerified,
      auto_verified_platforms: matchedPlatforms,
    });
  } catch (err) {
    console.error("Register error:", err);
    return NextResponse.json(
      { error: "Internal server error." },
      { status: 500 },
    );
  }
}
