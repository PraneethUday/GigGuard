'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'

const PLATFORMS = [
  { id: 'zomato', name: 'Zomato' },
  { id: 'swiggy', name: 'Swiggy' },
  { id: 'amazon', name: 'Amazon Flex' },
  { id: 'blinkit', name: 'Blinkit' },
  { id: 'zepto', name: 'Zepto' },
  { id: 'meesho', name: 'Meesho' },
  { id: 'porter', name: 'Porter' },
  { id: 'dunzo', name: 'Dunzo' },
]
const CITIES = ['Bengaluru', 'Chennai', 'Delhi', 'Mumbai', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Surat']
const STEPS = ['Platform', 'Personal Info', 'Documents', 'Coverage']

const inputStyle: React.CSSProperties = {
  width: '100%', height: 44, padding: '0 14px', fontSize: 14,
  border: '1px solid #d1d5db', borderRadius: 8, outline: 'none',
  transition: 'border-color 0.15s, box-shadow 0.15s',
}
const labelStyle: React.CSSProperties = {
  display: 'block', fontSize: 13, fontWeight: 600, color: '#374151', marginBottom: 6,
}
const focusHandler = {
  onFocus: (e: React.FocusEvent<HTMLInputElement | HTMLSelectElement>) => { e.target.style.borderColor = '#4f46e5'; e.target.style.boxShadow = '0 0 0 3px rgba(79,70,229,0.1)' },
  onBlur: (e: React.FocusEvent<HTMLInputElement | HTMLSelectElement>) => { e.target.style.borderColor = '#d1d5db'; e.target.style.boxShadow = 'none' },
}

type Form = {
  platforms: string[]; name: string; age: string; phone: string; email: string;
  password: string; confirmPassword: string;
  pan: string; aadhaar: string; city: string; area: string; deliveryId: string;
  upi: string; bank: string;
  consent: boolean; gpsConsent: boolean; autopay: boolean; tier: string;
}

export default function RegisterPage() {
  const router = useRouter()
  const [step, setStep] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [form, setForm] = useState<Form>({
    platforms: [], name: '', age: '', phone: '', email: '',
    password: '', confirmPassword: '',
    pan: '', aadhaar: '', city: '', area: '', deliveryId: '',
    upi: '', bank: '',
    consent: false, gpsConsent: false, autopay: false, tier: 'standard',
  })

  const update = (key: keyof Form, val: unknown) => setForm(p => ({ ...p, [key]: val }))
  const togglePlatform = (id: string) =>
    setForm(p => ({ ...p, platforms: p.platforms.includes(id) ? p.platforms.filter(x => x !== id) : [...p.platforms, id] }))
  const nextStep = () => { setError(''); setStep(s => Math.min(s + 1, 4)) }
  const prevStep = () => { setError(''); setStep(s => Math.max(s - 1, 1)) }

  const weeklyPremium = form.tier === 'basic' ? 30 : form.tier === 'standard' ? 60 : 105
  const maxPayout = form.tier === 'basic' ? 500 : form.tier === 'standard' ? 1200 : 2500

  const handleSubmit = async () => {
    if (form.password !== form.confirmPassword) { setError('Passwords do not match.'); return }
    if (form.password.length < 6) { setError('Password must be at least 6 characters.'); return }
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: form.name, age: form.age, phone: form.phone,
          email: form.email, password: form.password,
          city: form.city, area: form.area, deliveryId: form.deliveryId,
          platforms: form.platforms, pan: form.pan, aadhaar: form.aadhaar,
          upi: form.upi, bank: form.bank,
          consent: form.consent, gpsConsent: form.gpsConsent,
          autopay: form.autopay, tier: form.tier,
        }),
      })
      const data = await res.json()
      if (!res.ok) { setError(data.error || 'Registration failed.'); return }
      localStorage.setItem('gg_token', data.token)
      localStorage.setItem('gg_user', JSON.stringify(data.user))
      router.push('/dashboard')
    } catch {
      setError('Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const btnPrimary: React.CSSProperties = {
    height: 44, background: '#4f46e5', color: '#fff', border: 'none', borderRadius: 8,
    fontSize: 14, fontWeight: 600, cursor: 'pointer', transition: 'background 0.15s', flex: 2,
  }
  const btnOutline: React.CSSProperties = {
    height: 44, background: '#fff', color: '#374151', border: '1px solid #d1d5db', borderRadius: 8,
    fontSize: 14, fontWeight: 600, cursor: 'pointer', transition: 'background 0.15s', flex: 1,
  }

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', padding: '32px 24px' }}>
      <div style={{ maxWidth: 640, margin: '0 auto' }}>
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <div style={{
            width: 48, height: 48, background: '#4f46e5', borderRadius: 12,
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18, fontWeight: 800, color: '#fff', marginBottom: 12,
          }}>GG</div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0f172a', margin: 0 }}>Create your account</h1>
          <p style={{ color: '#64748b', fontSize: 14, marginTop: 4 }}>Register as a delivery partner on GigGuard</p>
        </div>

        {/* Stepper */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 32 }}>
          {STEPS.map((s, i) => (
            <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{
                width: 32, height: 32, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 13, fontWeight: 600,
                background: step > i + 1 ? '#4f46e5' : step === i + 1 ? '#4f46e5' : '#e2e8f0',
                color: step >= i + 1 ? '#fff' : '#64748b',
                transition: 'all 0.2s',
              }}>
                {step > i + 1 ? '\u2713' : i + 1}
              </div>
              <span style={{ fontSize: 13, fontWeight: step === i + 1 ? 600 : 400, color: step === i + 1 ? '#0f172a' : '#64748b', display: i < 3 ? undefined : undefined }}>
                {s}
              </span>
              {i < STEPS.length - 1 && <div style={{ width: 32, height: 2, background: step > i + 1 ? '#4f46e5' : '#e2e8f0', borderRadius: 1 }} />}
            </div>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8,
            padding: '10px 14px', marginBottom: 16, fontSize: 13, color: '#dc2626', fontWeight: 500,
          }}>{error}</div>
        )}

        {/* Card */}
        <div style={{
          background: '#fff', borderRadius: 16, padding: 32,
          boxShadow: '0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06)',
          border: '1px solid #e2e8f0',
        }}>

          {/* Step 1: Platforms */}
          {step === 1 && (
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>Choose your platforms</h2>
              <p style={{ color: '#64748b', fontSize: 14, marginBottom: 24 }}>Select all delivery platforms you work with.</p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 24 }}>
                {PLATFORMS.map(p => {
                  const sel = form.platforms.includes(p.id)
                  return (
                    <div key={p.id} onClick={() => togglePlatform(p.id)} style={{
                      padding: '16px 12px', textAlign: 'center', cursor: 'pointer', borderRadius: 10,
                      border: sel ? '2px solid #4f46e5' : '1px solid #e2e8f0',
                      background: sel ? '#eef2ff' : '#fff',
                      transition: 'all 0.15s',
                    }}>
                      <div style={{ fontSize: 14, fontWeight: 600, color: sel ? '#4f46e5' : '#374151' }}>{p.name}</div>
                      {sel && <div style={{ fontSize: 12, color: '#4f46e5', marginTop: 4, fontWeight: 500 }}>Selected</div>}
                    </div>
                  )
                })}
              </div>
              {form.platforms.length > 0 && (
                <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 8, padding: '10px 14px', marginBottom: 20, fontSize: 13, color: '#16a34a', fontWeight: 500 }}>
                  {form.platforms.length} platform{form.platforms.length > 1 ? 's' : ''} selected
                </div>
              )}
              <button disabled={form.platforms.length === 0} onClick={nextStep} style={{
                ...btnPrimary, width: '100%', flex: undefined,
                opacity: form.platforms.length === 0 ? 0.5 : 1,
                cursor: form.platforms.length === 0 ? 'not-allowed' : 'pointer',
              }}>Continue</button>
            </div>
          )}

          {/* Step 2: Personal Info */}
          {step === 2 && (
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 20 }}>Personal information</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={labelStyle}>Full Name</label>
                  <input style={inputStyle} placeholder="Your full name" value={form.name} onChange={e => update('name', e.target.value)} {...focusHandler} />
                </div>
                <div>
                  <label style={labelStyle}>Age</label>
                  <input style={inputStyle} placeholder="25" type="number" value={form.age} onChange={e => update('age', e.target.value)} {...focusHandler} />
                </div>
                <div>
                  <label style={labelStyle}>Phone</label>
                  <input style={inputStyle} placeholder="+91 98765 43210" value={form.phone} onChange={e => update('phone', e.target.value)} {...focusHandler} />
                </div>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={labelStyle}>Email</label>
                  <input style={inputStyle} type="email" placeholder="you@example.com" value={form.email} onChange={e => update('email', e.target.value)} {...focusHandler} />
                </div>
                <div>
                  <label style={labelStyle}>Password</label>
                  <input style={inputStyle} type="password" placeholder="Min. 6 characters" value={form.password} onChange={e => update('password', e.target.value)} {...focusHandler} />
                </div>
                <div>
                  <label style={labelStyle}>Confirm Password</label>
                  <input style={inputStyle} type="password" placeholder="Re-enter password" value={form.confirmPassword} onChange={e => update('confirmPassword', e.target.value)} {...focusHandler} />
                </div>
                <div>
                  <label style={labelStyle}>City</label>
                  <select style={{ ...inputStyle, cursor: 'pointer' }} value={form.city} onChange={e => update('city', e.target.value)} {...focusHandler}>
                    <option value="">Select city</option>
                    {CITIES.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Delivery Area / Zone</label>
                  <input style={inputStyle} placeholder="e.g. Koramangala" value={form.area} onChange={e => update('area', e.target.value)} {...focusHandler} />
                </div>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={labelStyle}>Primary Delivery Partner ID</label>
                  <input style={inputStyle} placeholder="Your ID from the platform app" value={form.deliveryId} onChange={e => update('deliveryId', e.target.value)} {...focusHandler} />
                  <p style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Used to verify your worker status on the selected platforms.</p>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 12, marginTop: 24 }}>
                <button onClick={prevStep} style={btnOutline}>Back</button>
                <button onClick={nextStep} style={btnPrimary}>Continue</button>
              </div>
            </div>
          )}

          {/* Step 3: Documents */}
          {step === 3 && (
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 20 }}>Identity & payment</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <label style={labelStyle}>PAN Card Number</label>
                  <input style={inputStyle} placeholder="ABCDE1234F" value={form.pan} onChange={e => update('pan', e.target.value.toUpperCase())} maxLength={10} {...focusHandler} />
                </div>
                <div>
                  <label style={labelStyle}>Aadhaar Number</label>
                  <input style={inputStyle} placeholder="1234 5678 9012" value={form.aadhaar} onChange={e => update('aadhaar', e.target.value)} maxLength={12} {...focusHandler} />
                </div>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={labelStyle}>UPI ID (for payouts)</label>
                  <input style={inputStyle} placeholder="yourname@upi" value={form.upi} onChange={e => update('upi', e.target.value)} {...focusHandler} />
                </div>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={labelStyle}>Bank Account (optional)</label>
                  <input style={inputStyle} placeholder="IFSC + Account Number" value={form.bank} onChange={e => update('bank', e.target.value)} {...focusHandler} />
                </div>
              </div>

              <div style={{ marginTop: 24, borderTop: '1px solid #e2e8f0', paddingTop: 20 }}>
                <p style={{ fontSize: 14, fontWeight: 600, color: '#374151', marginBottom: 14 }}>Consent & authorisations</p>
                {[
                  { key: 'consent' as keyof Form, label: 'I authorise GigGuard to monitor weather and disruption data in my delivery zone for insurance claims.' },
                  { key: 'gpsConsent' as keyof Form, label: 'I authorise GPS location validation during disruption events for fraud prevention.' },
                  { key: 'autopay' as keyof Form, label: 'Enable AutoPay — auto-deduct weekly premium from platform payout (5% discount).' },
                ].map(item => (
                  <label key={item.key} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, cursor: 'pointer', marginBottom: 12 }}>
                    <input type="checkbox" checked={!!form[item.key]} onChange={e => update(item.key, e.target.checked)}
                      style={{ width: 18, height: 18, accentColor: '#4f46e5', cursor: 'pointer', marginTop: 2, flexShrink: 0 }} />
                    <span style={{ fontSize: 13, color: '#374151', lineHeight: 1.5 }}>{item.label}</span>
                  </label>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 12, marginTop: 24 }}>
                <button onClick={prevStep} style={btnOutline}>Back</button>
                <button onClick={nextStep} style={btnPrimary}>Continue</button>
              </div>
            </div>
          )}

          {/* Step 4: Coverage */}
          {step === 4 && (
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 20 }}>Choose your coverage</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 24 }}>
                {[
                  { id: 'basic', tier: 'Basic', price: '20-40', payout: '500' },
                  { id: 'standard', tier: 'Standard', price: '40-80', payout: '1,200', recommended: true },
                  { id: 'pro', tier: 'Pro', price: '80-130', payout: '2,500' },
                ].map(t => {
                  const sel = form.tier === t.id
                  return (
                    <div key={t.id} onClick={() => update('tier', t.id)} style={{
                      padding: 20, cursor: 'pointer', borderRadius: 12, textAlign: 'center',
                      border: sel ? '2px solid #4f46e5' : '1px solid #e2e8f0',
                      background: sel ? '#eef2ff' : '#fff', transition: 'all 0.15s', position: 'relative',
                    }}>
                      {t.recommended && <div style={{ position: 'absolute', top: -10, left: '50%', transform: 'translateX(-50%)', background: '#4f46e5', color: '#fff', fontSize: 10, fontWeight: 700, padding: '2px 10px', borderRadius: 10, whiteSpace: 'nowrap' }}>Recommended</div>}
                      <div style={{ fontSize: 15, fontWeight: 700, color: sel ? '#4f46e5' : '#0f172a', marginBottom: 4 }}>{t.tier}</div>
                      <div style={{ fontSize: 22, fontWeight: 800, color: sel ? '#4f46e5' : '#0f172a' }}>&#8377;{t.price}</div>
                      <div style={{ fontSize: 12, color: '#64748b' }}>per week</div>
                      <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>Max payout: &#8377;{t.payout}</div>
                    </div>
                  )
                })}
              </div>

              {/* Summary */}
              <div style={{ background: '#f8fafc', borderRadius: 10, padding: 20, marginBottom: 24, border: '1px solid #e2e8f0' }}>
                <p style={{ fontSize: 14, fontWeight: 600, color: '#374151', marginBottom: 12 }}>Summary</p>
                {[
                  ['Platforms', form.platforms.map(id => PLATFORMS.find(p => p.id === id)?.name).join(', ') || '-'],
                  ['Weekly Premium (est.)', `\u20B9${weeklyPremium}`],
                  ['Max Weekly Payout', `\u20B9${maxPayout}`],
                  ['AutoPay', form.autopay ? 'Enabled (5% discount)' : 'Disabled'],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #e2e8f0', fontSize: 13 }}>
                    <span style={{ color: '#64748b' }}>{k}</span>
                    <span style={{ fontWeight: 600, color: '#0f172a' }}>{v}</span>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', gap: 12 }}>
                <button onClick={prevStep} style={btnOutline}>Back</button>
                <button onClick={handleSubmit} disabled={loading} style={{ ...btnPrimary, opacity: loading ? 0.7 : 1, cursor: loading ? 'not-allowed' : 'pointer' }}>
                  {loading ? 'Creating account...' : 'Create account'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <p style={{ textAlign: 'center', fontSize: 14, color: '#64748b', marginTop: 24 }}>
          Already have an account?{' '}
          <span style={{ color: '#4f46e5', fontWeight: 600, cursor: 'pointer' }} onClick={() => router.push('/login')}>Sign in</span>
        </p>
      </div>
    </div>
  )
}
