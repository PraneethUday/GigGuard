import { createClient } from '@supabase/supabase-js'

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY!

// Server-side client using service role key (API routes only — never expose to browser)
export function createServerClient() {
  return createClient(SUPABASE_URL, SERVICE_KEY)
}

// Maps frontend platform id → Supabase table name
export const PLATFORM_TABLE: Record<string, string> = {
  swiggy:      'swiggy_workers',
  zomato:      'zomato_workers',
  amazon:      'amazon_flex_workers',
  amazon_flex: 'amazon_flex_workers',
  blinkit:     'blinkit_workers',
  zepto:       'zepto_workers',
  meesho:      'meesho_workers',
  porter:      'porter_workers',
  dunzo:       'dunzo_workers',
}
