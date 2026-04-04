/**
 * In-memory payment store shared across Next.js API routes.
 * Payments are accumulated here for the lifetime of the server process.
 * Both web and mobile payments are written here via /api/payment/pay.
 */

export interface PaymentRecord {
  transaction_id: string;
  amount: number;
  method: string;
  tier: string;
  worker_id: string;
  status: string;
  timestamp: string;
}

// Module-level store: Map<worker_id, PaymentRecord[]> (newest first)
const store = new Map<string, PaymentRecord[]>();

export function addPayment(record: PaymentRecord): void {
  const existing = store.get(record.worker_id) ?? [];
  store.set(record.worker_id, [record, ...existing]);
}

export function getPayments(workerId: string): PaymentRecord[] {
  return store.get(workerId) ?? [];
}
