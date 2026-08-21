import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, getHealth } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("API client", () => {
  it("wraps malformed successful JSON in ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("not-json", { status: 200 })));

    await expect(getHealth()).rejects.toMatchObject({
      name: "ApiError",
      status: 200,
      code: "INVALID_RESPONSE",
    } satisfies Partial<ApiError>);
  });
});
