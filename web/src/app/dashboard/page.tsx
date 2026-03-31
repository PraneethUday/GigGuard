'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'

type User = {
  id: string; name: string; email: string; platforms: string[];
  tier: string; verification_status: string; city: string;
  area?: string; delivery_id?: string; autopay?: boolean; phone?: string;
}

const PLATFORM_NAMES: Record<string, string> = {
  swiggy: 'Swiggy', zomato: 'Zomato', amazon: 'Amazon Flex',
  blinkit: 'Blinkit', zepto: 'Zepto', meesho: 'Meesho',
  porter: 'Porter', dunzo: 'Dunzo',
}

const NAV = [
  { id: 'home', label: 'Dashboard' },
  { id: 'profile', label: 'Profile' },
]

export default function DashboardPage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [tab, setTab] = useState('home')

  useEffect(() => {
    const token = localStorage.getItem('gg_token')
    const raw = localStorage.getItem('gg_user')
    if (!token || !raw) { router.replace('/login'); return }
    try { setUser(JSON.parse(raw)) } catch { router.replace('/login') }
  }, [router])

  const logout = () => {
    localStorage.removeItem('gg_token')
    localStorage.removeItem('gg_user')
    router.push('/login')
  }

  if (!user) return null

  const tierLabel = user.tier === 'basic' ? 'Basic Shield' : user.tier === 'pro' ? 'Pro Protect' : 'Standard Guard'
  const verified = user.verification_status === 'verified'

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header style={{
        background: '#fff', borderBottom: '1px solid #e2e8f0', padding: '0 24px', height: 60,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 36, height: 36, background: '#4f46e5', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, fontWeight: 800, color: '#fff' }}>GG</div>
          <span style={{ fontSize: 17, fontWeight: 700, color: '#0f172a' }}>GigGuard</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 32, height: 32, borderRadius: '50%', background: '#eef2ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, fontWeight: 700, color: '#4f46e5' }}>
              {user.name.charAt(0).toUpperCase()}
            </div>
            <span style={{ fontSize: 14, fontWeight: 600, color: '#0f172a' }}>{user.name}</span>
          </div>
          <button onClick={logout} style={{
            background: '#fff', border: '1px solid #d1d5db', borderRadius: 8, padding: '6px 14px',
            fontSize: 13, fontWeight: 500, color: '#374151', cursor: 'pointer',
          }}>Logout</button>
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1 }}>
        {/* Sidebar */}
        <aside style={{ width: 220, background: '#fff', borderRight: '1px solid #e2e8f0', padding: '20px 0' }}>
          {NAV.map(item => (
            <button key={item.id} onClick={() => setTab(item.id)} style={{
              display: 'block', width: '100%', padding: '10px 20px', border: 'none', textAlign: 'left',
              background: tab === item.id ? '#eef2ff' : 'transparent',
              color: tab === item.id ? '#4f46e5' : '#374151',
              fontWeight: tab === item.id ? 600 : 500, fontSize: 14, cursor: 'pointer',
              borderLeft: tab === item.id ? '3px solid #4f46e5' : '3px solid transparent',
              transition: 'all 0.12s',
            }}>{item.label}</button>
          ))}
        </aside>

        {/* Main */}
        <main style={{ flex: 1, padding: 32, overflowY: 'auto' }}>
          <div style={{ maxWidth: 900, margin: '0 auto' }}>

            {tab === 'home' && (
              <div>
                <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Welcome, {user.name}</h1>
                <p style={{ color: '#64748b', fontSize: 14, marginBottom: 28 }}>Here&apos;s your account overview.</p>

                {/* Status cards */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 28 }}>
                  <Card label="Verification" value={verified ? 'Verified' : 'Pending'} color={verified ? '#16a34a' : '#d97706'} />
                  <Card label="Coverage Tier" value={tierLabel} color="#4f46e5" />
                  <Card label="City" value={user.city || '-'} color="#0f172a" />
                </div>

                {/* Connected platforms */}
                <div style={{ background: '#fff', borderRadius: 12, border: '1px solid #e2e8f0', padding: 24, marginBottom: 24 }}>
                  <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Connected platforms</h3>
                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                    {user.platforms.map(p => (
                      <span key={p} style={{
                        background: '#eef2ff', color: '#4f46e5', padding: '6px 14px', borderRadius: 8,
                        fontSize: 13, fontWeight: 600,
                      }}>{PLATFORM_NAMES[p] || p}</span>
                    ))}
                  </div>
                </div>

                {/* Info banner */}
                {!verified && (
                  <div style={{
                    background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 10,
                    padding: '14px 18px', fontSize: 13, color: '#92400e',
                  }}>
                    Your account is pending verification. Your delivery partner ID is being validated against platform records.
                  </div>
                )}
              </div>
            )}

            {tab === 'profile' && (
              <div>
                <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 24 }}>Profile</h1>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                  <div style={{ background: '#fff', borderRadius: 12, border: '1px solid #e2e8f0', padding: 24 }}>
                    <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Personal information</h3>
                    <ProfileRow label="Name" value={user.name} />
                    <ProfileRow label="Email" value={user.email} />
                    <ProfileRow label="Phone" value={user.phone || '-'} />
                    <ProfileRow label="City" value={user.city || '-'} />
                    <ProfileRow label="Area" value={user.area || '-'} />
                  </div>
                  <div style={{ background: '#fff', borderRadius: 12, border: '1px solid #e2e8f0', padding: 24 }}>
                    <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Account details</h3>
                    <ProfileRow label="Delivery ID" value={user.delivery_id || '-'} />
                    <ProfileRow label="Coverage Tier" value={tierLabel} />
                    <ProfileRow label="Verification" value={verified ? 'Verified' : 'Pending'} />
                    <ProfileRow label="AutoPay" value={user.autopay ? 'Enabled' : 'Disabled'} />
                    <ProfileRow label="Platforms" value={user.platforms.map(p => PLATFORM_NAMES[p] || p).join(', ')} />
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

function Card({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ background: '#fff', borderRadius: 12, border: '1px solid #e2e8f0', padding: 20 }}>
      <div style={{ fontSize: 12, color: '#64748b', fontWeight: 500, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color }}>{value}</div>
    </div>
  )
}

function ProfileRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid #f1f5f9', fontSize: 14 }}>
      <span style={{ color: '#64748b' }}>{label}</span>
      <span style={{ fontWeight: 600, color: '#0f172a' }}>{value}</span>
    </div>
  )
}
