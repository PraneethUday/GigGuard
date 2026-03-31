import { NextRequest, NextResponse } from 'next/server'
import { createServerClient } from '@/lib/supabase'
import { verifyPassword, createToken } from '@/lib/auth'

export async function POST(req: NextRequest) {
  try {
    const { email, password } = await req.json()

    if (!email || !password) {
      return NextResponse.json({ error: 'Email and password are required.' }, { status: 400 })
    }

    const db = createServerClient()

    // Find user by email
    const { data: users } = await db
      .from('registered_workers')
      .select('id, name, email, phone, password_hash, platforms, tier, verification_status, city, area, delivery_id, autopay')
      .eq('email', email)
      .eq('is_active', true)
      .limit(1)

    if (!users || users.length === 0) {
      return NextResponse.json({ error: 'Invalid email or password.' }, { status: 401 })
    }

    const user = users[0]
    const valid = await verifyPassword(password, user.password_hash)

    if (!valid) {
      return NextResponse.json({ error: 'Invalid email or password.' }, { status: 401 })
    }

    const token = await createToken(user.id)

    // Return user without password hash
    const { password_hash: _, ...safeUser } = user

    return NextResponse.json({ token, user: safeUser })
  } catch (err) {
    console.error('Login error:', err)
    return NextResponse.json({ error: 'Internal server error.' }, { status: 500 })
  }
}
