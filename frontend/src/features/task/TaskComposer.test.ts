import { describe, expect, it } from "vitest";

import { parseNumberDraft } from "./TaskComposer";

describe("number input drafts", () => {
  it("keeps empty and exponent intermediate states out of committed values", () => {
    expect(parseNumberDraft("")).toBeNull();
    expect(parseNumberDraft("1e")).toBeNull();
    expect(parseNumberDraft("0")).toBe(0);
    expect(parseNumberDraft("12.5")).toBe(12.5);
  });
});
