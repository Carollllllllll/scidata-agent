import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError, createAnalysisTask, createDiscoveryTask } from "../../api/client";
import { Icon } from "../../components/Icon";
import { formatBytes } from "../../lib/task";

const ACCEPTED_EXTENSIONS = [".pdf", ".csv", ".tsv", ".xlsx", ".xls"];
const MAX_FILE_BYTES = 50 * 1024 * 1024;

const promptExamples = [
  "研究 Ia 型超新星光变曲线，整合峰值星等、衰减率和来源证据",
  "比较钙钛矿太阳能电池的材料、制备方法、PCE 与稳定性",
  "整理虚拟试穿论文的数据集、指标、分辨率和实验设置",
];

export function TaskComposer() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepthRef = useRef(0);
  const [mode, setMode] = useState<"analysis" | "discovery">("analysis");
  const [question, setQuestion] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxPdfPages, setMaxPdfPages] = useState(8);
  const [maxAutoResources, setMaxAutoResources] = useState(5);
  const [enableLiveSearch, setEnableLiveSearch] = useState(true);
  const [autoDownloadSources, setAutoDownloadSources] = useState(true);
  const [maxDynamicTextBlocks, setMaxDynamicTextBlocks] = useState(20);
  const [maxRecordTextBlocks, setMaxRecordTextBlocks] = useState(20);
  const [maxFiguresPerPdf, setMaxFiguresPerPdf] = useState(6);
  const [reuseDynamicRecordsForMetrics, setReuseDynamicRecordsForMetrics] = useState(true);

  const mutation = useMutation({
    mutationFn: () => {
      if (mode === "discovery") return createDiscoveryTask(question);
      setUploadProgress(0);
      return createAnalysisTask(
        {
          researchQuestion: question,
          files,
          maxPdfPages,
          maxArxivPapers: null,
          maxAutoResources,
          enableLiveSearch,
          autoDownloadSources,
          maxDynamicTextBlocks,
          maxRecordTextBlocks,
          maxFiguresPerPdf,
          reuseDynamicRecordsForMetrics,
        },
        setUploadProgress,
      );
    },
    onSuccess: async (task) => {
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
      navigate(`/tasks/${task.task_id}`);
    },
    onError: () => setUploadProgress(null),
  });

  function addFiles(incoming: File[]) {
    setFormError(null);
    const invalidType = incoming.find(
      (file) => !ACCEPTED_EXTENSIONS.some((extension) => file.name.toLowerCase().endsWith(extension)),
    );
    if (invalidType) {
      setFormError(`不支持 ${invalidType.name}；请选择 PDF、CSV、TSV 或 Excel 文件。`);
      return;
    }
    const oversized = incoming.find((file) => file.size > MAX_FILE_BYTES);
    if (oversized) {
      setFormError(`${oversized.name} 超过 50 MB 单文件限制。`);
      return;
    }
    const known = new Set(files.map((file) => `${file.name}:${file.size}:${file.lastModified}`));
    const additions = incoming.filter((file) => !known.has(`${file.name}:${file.size}:${file.lastModified}`));
    if (files.length + additions.length > 20) {
      setFormError(`每个任务最多上传 20 个文件；本次仅加入前 ${Math.max(0, 20 - files.length)} 个。`);
    }
    setFiles([...files, ...additions].slice(0, 20));
  }

  function submit(event: React.FormEvent) {
    event.preventDefault();
    setFormError(null);
    if (question.trim().length < 3) {
      setFormError("请描述一个明确的科研问题或数据需求。");
      return;
    }
    mutation.mutate();
  }

  const mutationError = mutation.error instanceof ApiError ? mutation.error.message : null;

  return (
    <form className="composer-card" onSubmit={submit}>
      <div className="composer-glow" />
      <div className="composer-header">
        <div>
          <span className="eyebrow"><Icon name="spark" size={15} /> NEW RESEARCH TASK</span>
          <h2>从科研问题，到可验证的数据</h2>
          <p>描述目标，Agent 将规划字段、查找来源、解析资料并保留每条数据的证据链。</p>
        </div>
        <div className="mode-switch" role="group" aria-label="任务模式">
          <button type="button" className={mode === "analysis" ? "active" : ""} onClick={() => setMode("analysis")}>
            完整分析
          </button>
          <button type="button" className={mode === "discovery" ? "active" : ""} onClick={() => setMode("discovery")}>
            仅发现来源
          </button>
        </div>
      </div>

      <label className="question-field">
        <span className="sr-only">科研问题</span>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="例如：我希望研究 Ia 型超新星光变曲线，请整合相关论文、开放数据库和表格中的关键参数，并标注每项数据来源……"
          rows={5}
          maxLength={4000}
          disabled={mutation.isPending}
        />
        <span className="question-count">{question.length} / 4000</span>
      </label>

      <div className="prompt-examples">
        <span>试试这样问</span>
        {promptExamples.map((prompt) => (
          <button type="button" key={prompt} onClick={() => setQuestion(prompt)}>
            {prompt}
          </button>
        ))}
      </div>

      {mode === "analysis" && (
        <>
          <input
            ref={inputRef}
            hidden
            type="file"
            accept={ACCEPTED_EXTENSIONS.join(",")}
            multiple
            onChange={(event) => {
              addFiles(Array.from(event.target.files ?? []));
              event.currentTarget.value = "";
            }}
          />
          <button
            className={`dropzone${dragging ? " dragging" : ""}`}
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragEnter={(event) => {
              event.preventDefault();
              dragDepthRef.current += 1;
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => {
              event.preventDefault();
              dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
              if (dragDepthRef.current === 0) setDragging(false);
            }}
            onDrop={(event) => {
              event.preventDefault();
              dragDepthRef.current = 0;
              setDragging(false);
              addFiles(Array.from(event.dataTransfer.files));
            }}
          >
            <span className="dropzone-icon"><Icon name="upload" size={22} /></span>
            <span><strong>上传论文与数据文件</strong><small>拖拽至此，或点击选择 · PDF / CSV / TSV / Excel · 单文件 ≤ 50 MB</small></span>
            <span className="dropzone-action">选择文件</span>
          </button>

          {files.length > 0 && (
            <div className="selected-files">
              {files.map((file, index) => (
                <div className="selected-file" key={`${file.name}:${file.size}:${file.lastModified}`}>
                  <span className="file-kind"><Icon name="file" size={16} /></span>
                  <span><strong>{file.name}</strong><small>{formatBytes(file.size)}</small></span>
                  <button type="button" aria-label={`移除 ${file.name}`} onClick={() => setFiles((items) => items.filter((_, itemIndex) => itemIndex !== index))}>
                    <Icon name="close" size={15} />
                  </button>
                </div>
              ))}
            </div>
          )}

          <button className="advanced-toggle" type="button" onClick={() => setShowAdvanced((value) => !value)}>
            <Icon name="settings" size={16} />
            分析参数
            <Icon name="chevron" size={14} className={showAdvanced ? "rotate" : ""} />
          </button>
          {showAdvanced && (
            <div className="advanced-grid">
              <NumberField label="每篇 PDF 页数" value={maxPdfPages} min={1} max={200} onChange={setMaxPdfPages} />
              <NumberField label="自动获取资源数" value={maxAutoResources} min={0} max={100} onChange={setMaxAutoResources} />
              <NumberField label="动态文本块" value={maxDynamicTextBlocks} min={1} max={500} onChange={setMaxDynamicTextBlocks} />
              <NumberField label="指标文本块" value={maxRecordTextBlocks} min={1} max={500} onChange={setMaxRecordTextBlocks} />
              <NumberField label="每篇图表数" value={maxFiguresPerPdf} min={0} max={50} onChange={setMaxFiguresPerPdf} />
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={enableLiveSearch}
                  onChange={(event) => {
                    setEnableLiveSearch(event.target.checked);
                    if (!event.target.checked) setAutoDownloadSources(false);
                  }}
                />
                <span><strong>联网查找多源资料</strong><small>即使已上传文件，也会继续查询论文与开放数据源</small></span>
              </label>
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={reuseDynamicRecordsForMetrics}
                  onChange={(event) => setReuseDynamicRecordsForMetrics(event.target.checked)}
                />
                <span><strong>复用动态抽取结果</strong><small>避免对同一正文重复调用指标抽取模型；没有数值时会自动回退</small></span>
              </label>
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={autoDownloadSources}
                  disabled={!enableLiveSearch}
                  onChange={(event) => setAutoDownloadSources(event.target.checked)}
                />
                <span><strong>自动下载选中资料</strong><small>关闭后仅保留检索元数据，不下载远程文件</small></span>
              </label>
            </div>
          )}
        </>
      )}

      {(formError || mutationError) && (
        <div className="form-error"><Icon name="warning" size={17} /> {formError ?? mutationError}</div>
      )}

      {mutation.isPending && mode === "analysis" && uploadProgress !== null && (
        <div className="upload-progress" aria-live="polite">
          <div><span>正在上传与创建任务</span><strong>{uploadProgress}%</strong></div>
          <progress max={100} value={uploadProgress} />
        </div>
      )}

      <div className="composer-footer">
        <div className="privacy-copy"><Icon name="shield" size={16} /> 文件仅交给当前后端任务处理，不在浏览器调用模型。</div>
        <button className="primary-button" type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? <><span className="spinner" /> {uploadProgress !== null && uploadProgress < 100 ? `正在上传 ${uploadProgress}%` : "正在创建任务"}</> : <>{mode === "analysis" ? "开始分析" : "发现数据来源"}<Icon name="arrow" size={18} /></>}
        </button>
      </div>
    </form>
  );
}

export function parseNumberDraft(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => setDraft(String(value)), [value]);

  return (
    <label className="number-field">
      <span>{label}</span>
      <input
        type="number"
        value={draft}
        min={min}
        max={max}
        step={1}
        onChange={(event) => {
          const next = event.target.value;
          setDraft(next);
          const parsed = parseNumberDraft(next);
          if (parsed !== null && parsed >= min && parsed <= max) onChange(parsed);
        }}
        onBlur={() => {
          const parsed = parseNumberDraft(draft);
          const committed = parsed === null ? value : Math.max(min, Math.min(max, parsed));
          onChange(committed);
          setDraft(String(committed));
        }}
      />
    </label>
  );
}
