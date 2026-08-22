import { NextRequest, NextResponse } from "next/server";

const SYSTEM_PROMPT = `You are WPIP Assistant, a helpful AI for the WPIP (Worker Protection Insurance Platform) app — a weekly insurance platform for gig delivery workers in India (Zomato, Swiggy, Blinkit, Zepto, Amazon Flex, Meesho, Porter, Dunzo).

You help workers understand:
- How WPIP insurance works: claims trigger automatically when verified weather disruptions (flood, extreme heat, heavy rain, cyclone, hailstorm), curfews, or other major disruptions are detected in their registered city.
- No manual filing needed — the system detects events and files claims on behalf of the worker.
- Three coverage tiers: Basic (₹30/week base, max ₹500 payout), Standard (₹60/week base, max ₹1,200 payout), Pro (₹105/week base, max ₹2,500 payout). Premiums are adjusted weekly based on city weather risk and AQI.
- AutoPay gives a 5% discount on premiums, deducted from platform payout.
- Claims are settled within 24–48 hours via UPI after approval.
- Workers must pay the weekly premium to be covered. Coverage is not valid if the premium is unpaid for that week.
- Only claims from the worker's registration date onwards count — prior history is not considered.
- The worker's delivery city must match their registered platform city.
- Workers can check claim status, payment history, and weather risk in the app.
- Fraud detection automatically flags suspicious claim patterns.

Keep responses concise, friendly, and in plain English. If the user writes in Hindi or another Indian language, respond in that language. Do not make up claim amounts or specific policy details beyond what is listed above.`;

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json();
    if (!Array.isArray(messages) || messages.length === 0) {
      return NextResponse.json({ error: "No messages provided." }, { status: 400 });
    }

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: "AI service not configured." }, { status: 500 });
    }

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          ...messages.slice(-12), // keep last 12 messages for context
        ],
        max_tokens: 400,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      console.error("OpenAI error:", err);
      return NextResponse.json({ error: "AI service unavailable." }, { status: 502 });
    }

    const data = await response.json();
    const reply = data.choices?.[0]?.message?.content ?? "Sorry, I couldn't generate a response.";
    return NextResponse.json({ reply });
  } catch (err) {
    console.error("Chat route error:", err);
    return NextResponse.json({ error: "Something went wrong." }, { status: 500 });
  }
}
