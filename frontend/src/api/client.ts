import type {
  AnalyzeOptions,
  HealthResponse,
  TaskEventsResponse,
  TaskListResponse,
  TaskResponse,
  TaskSubmissionResponse,
  ReviewDecision,
} from "../types/api";

const configuredBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() ?? "";
export const API_BASE_URL = configuredBaseUrl.replace(/\/$/, "");
const configuredApiToken = (import.meta.env.VITE_SCIDATA_API_TOKEN as string | undefined)?.trim() ?? "";
export const API_AUTH_ENABLED = Boolean(configuredApiToken);

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalized}`;
}

function authorizedHeaders(initial?: HeadersInit): Headers {
  const headers = new Headers(initial);
  if (configuredApiToken) headers.set("Authorization", `Bearer ${configuredApiToken}`);
  return headers;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    const headers = authorizedHeaders(init?.headers);
    response = await fetch(apiUrl(path), { ...init, headers });
  } catch {
    throw new ApiError("无法连接后端服务，请确认 API 已启动。", 0, "NETWORK_ERROR");
  }

  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`;
    let code: string | undefined;
    try {
      const payload = (await response.json()) as {
        detail?: string | { message?: string; code?: string };
      };
      if (typeof payload.detail === "string") message = payload.detail;
      if (payload.detail && typeof payload.detail === "object") {
        message = payload.detail.message ?? message;
        code = payload.detail.code;
      }
    } catch {
      // Keep the HTTP fallback message for non-JSON responses.
    }
    throw new ApiError(message, response.status, code);
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(
      "后端返回了无法解析的响应，请稍后重试或检查服务日志。",
      response.status,
      "INVALID_RESPONSE",
    );
  }
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health");
}

export function listTasks(limit = 12): Promise<TaskListResponse> {
  return requestJson<TaskListResponse>(`/api/tasks?limit=${limit}`);
}

export function getTask(taskId: string): Promise<TaskResponse> {
  return requestJson<TaskResponse>(`/api/tasks/${encodeURIComponent(taskId)}`);
}

export function cancelTask(taskId: string): Promise<TaskResponse> {
  return requestJson<TaskResponse>(`/api/tasks/${encodeURIComponent(taskId)}/cancel`, { method: "POST" });
}

export function retryTask(taskId: string): Promise<TaskSubmissionResponse> {
  return requestJson<TaskSubmissionResponse>(`/api/tasks/${encodeURIComponent(taskId)}/retry`, { method: "POST" });
}

export function resumeTask(taskId: string): Promise<TaskSubmissionResponse> {
  return requestJson<TaskSubmissionResponse>(`/api/tasks/${encodeURIComponent(taskId)}/resume`, { method: "POST" });
}

export function getTaskEvents(taskId: string, tail = 60): Promise<TaskEventsResponse> {
  return requestJson<TaskEventsResponse>(
    `/api/tasks/${encodeURIComponent(taskId)}/events?tail=${tail}`,
  );
}

export async function getTextAsset(path: string): Promise<string> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), { headers: authorizedHeaders() });
  } catch {
    throw new ApiError("无法读取调研报告，请确认 API 已启动。", 0, "NETWORK_ERROR");
  }
  if (!response.ok) {
    throw new ApiError(`调研报告读取失败（HTTP ${response.status}）`, response.status);
  }
  return response.text();
}

export async function getAssetBlob(path: string): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), { headers: authorizedHeaders() });
  } catch {
    throw new ApiError("无法读取文件，请确认 API 已启动。", 0, "NETWORK_ERROR");
  }
  if (!response.ok) {
    throw new ApiError(`文件读取失败（HTTP ${response.status}）`, response.status);
  }
  return response.blob();
}

export async function openApiAsset(
  path: string,
  options: { download?: boolean; filename?: string } = {},
): Promise<void> {
  const previewWindow = options.download ? null : window.open("about:blank", "_blank");
  try {
    const blob = await getAssetBlob(path);
    const objectUrl = URL.createObjectURL(blob);
    if (previewWindow) {
      previewWindow.location.href = objectUrl;
    } else {
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = options.filename ?? "";
      if (!options.download) {
        anchor.target = "_blank";
        anchor.rel = "noreferrer";
      }
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    }
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
  } catch (error) {
    previewWindow?.close();
    throw error;
  }
}

export function submitReview(
  taskId: string,
  recordId: string,
  decision: ReviewDecision["decision"],
  note?: string,
): Promise<ReviewDecision> {
  return requestJson<ReviewDecision>(
    `/api/tasks/${encodeURIComponent(taskId)}/reviews/${encodeURIComponent(recordId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, note: note?.trim() || null }),
    },
  );
}

export function createAnalysisTask(
  options: AnalyzeOptions,
  onUploadProgress?: (percent: number) => void,
): Promise<TaskSubmissionResponse> {
  const form = new FormData();
  form.set("research_question", options.researchQuestion.trim());
  form.set("max_pdf_pages", String(options.maxPdfPages));
  if (options.maxArxivPapers !== null) {
    form.set("max_arxiv_papers", String(options.maxArxivPapers));
  }
  form.set("max_auto_resources", String(options.maxAutoResources));
  form.set("enable_live_search", String(options.enableLiveSearch));
  form.set("auto_download_sources", String(options.autoDownloadSources));
  form.set("max_dynamic_text_blocks", String(options.maxDynamicTextBlocks));
  form.set("max_record_text_blocks", String(options.maxRecordTextBlocks));
  form.set("max_figures_per_pdf", String(options.maxFiguresPerPdf));
  form.set("reuse_dynamic_records_for_metrics", String(options.reuseDynamicRecordsForMetrics));
  options.files.forEach((file) => form.append("files", file));
  return requestFormWithProgress<TaskSubmissionResponse>("/api/analyze", form, onUploadProgress);
}

function requestFormWithProgress<T>(
  path: string,
  form: FormData,
  onUploadProgress?: (percent: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", apiUrl(path));
    if (configuredApiToken) request.setRequestHeader("Authorization", `Bearer ${configuredApiToken}`);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable || event.total <= 0) return;
      onUploadProgress?.(Math.min(100, Math.round((event.loaded / event.total) * 100)));
    });
    request.addEventListener("error", () => {
      reject(new ApiError("无法连接后端服务，请确认 API 已启动。", 0, "NETWORK_ERROR"));
    });
    request.addEventListener("abort", () => {
      reject(new ApiError("上传已取消。", 0, "UPLOAD_ABORTED"));
    });
    request.addEventListener("load", () => {
      let payload: unknown;
      try {
        payload = JSON.parse(request.responseText);
      } catch {
        reject(
          new ApiError(
            request.status >= 200 && request.status < 300
              ? "后端返回了无法解析的响应，请稍后重试或检查服务日志。"
              : `请求失败（HTTP ${request.status}）`,
            request.status,
            request.status >= 200 && request.status < 300 ? "INVALID_RESPONSE" : undefined,
          ),
        );
        return;
      }
      if (request.status < 200 || request.status >= 300) {
        const detail = (payload as { detail?: string | { message?: string; code?: string } })?.detail;
        const message =
          typeof detail === "string"
            ? detail
            : detail?.message ?? `请求失败（HTTP ${request.status}）`;
        reject(new ApiError(message, request.status, typeof detail === "object" ? detail.code : undefined));
        return;
      }
      onUploadProgress?.(100);
      resolve(payload as T);
    });
    request.send(form);
  });
}

export function createDiscoveryTask(researchQuestion: string): Promise<TaskSubmissionResponse> {
  const form = new FormData();
  form.set("research_question", researchQuestion.trim());
  return requestJson<TaskSubmissionResponse>("/api/discover", {
    method: "POST",
    body: form,
  });
}
