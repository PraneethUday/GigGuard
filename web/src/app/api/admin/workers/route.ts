import { NextResponse } from 'next/server'
import { createServerClient } from '@/lib/supabase'

export async function GET() {
  try {
    const db = createServerClient()

    const { data, error } = await db
      .from('registered_workers')
      .select('id, name, email, phone, city, area, platforms, tier, verification_status, delivery_id, autopay, is_active, created_at')
      .order('created_at', { ascending: false })
      .limit(100)

    if (error) {
      console.error('Admin fetch error:', error)
      return NextResponse.json({ error: 'Failed to fetch workers.' }, { status: 500 })
    }

    return NextResponse.json({ workers: data })
  } catch (err) {
    console.error('Admin workers error:', err)
    return NextResponse.json({ error: 'Internal server error.' }, { status: 500 })
  }
}
