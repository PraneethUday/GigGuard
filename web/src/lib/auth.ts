import { SignJWT, jwtVerify } from "jose";
import bcrypt from "bcryptjs";

const primarySecret = process.env.JWT_SECRET || "wpip-dev-secret";
const legacyDevSecret = "wpip-dev-secret";

const signingSecret = new TextEncoder().encode(primarySecret);

const verifySecrets = [
  new TextEncoder().encode(primarySecret),
  ...(primarySecret !== legacyDevSecret
    ? [new TextEncoder().encode(legacyDevSecret)]
    : []),
];

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 12);
}

export async function verifyPassword(
  password: string,
  hash: string,
): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

export async function createToken(userId: string): Promise<string> {
  return new SignJWT({ sub: userId })
    .setProtectedHeader({ alg: "HS256" })
    .setExpirationTime("30d")
    .sign(signingSecret);
}

export async function verifyToken(token: string): Promise<string | null> {
  for (const secret of verifySecrets) {
    try {
      const { payload } = await jwtVerify(token, secret);
      if (payload.sub) return payload.sub as string;
    } catch {
      // try next known secret
    }
  }

  return null;
}
