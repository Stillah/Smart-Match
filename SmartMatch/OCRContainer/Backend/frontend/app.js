const state = {
  ocrConfig: null,
  trainingModels: [],
  selectedStages: [],
  selectedTrainingModel: null,
  jobsPollHandle: null,
};

const elements = {
  tabButtons: Array.from(document.querySelectorAll(".tab-button")),
  tabPanels: Array.from(document.querySelectorAll(".tab-panel")),
  ocrStageList: document.getElementById("ocr-stage-list"),
  ocrHandwrittenModel: document.getElementById("ocr-handwritten-model"),
  ocrManualLabelRow: document.getElementById("ocr-manual-label-row"),
  ocrManualLabel: document.getElementById("ocr-manual-label"),
  ocrFileRow: document.getElementById("ocr-file-row"),
  ocrFiles: document.getElementById("ocr-files"),
  ocrTextRow: document.getElementById("ocr-text-row"),
  ocrText: document.getElementById("ocr-text"),
  ocrInputHelp: document.getElementById("ocr-input-help"),
  ocrProcessButton: document.getElementById("ocr-process-button"),
  ocrRunButton: document.getElementById("ocr-run-button"),
  ocrStatus: document.getElementById("ocr-status"),
  ocrStructuredResult: document.getElementById("ocr-structured-result"),
  ocrResults: document.getElementById("ocr-results"),
  trainingModelSelect: document.getElementById("training-model-select"),
  trainingModelStatus: document.getElementById("training-model-status"),
  trainingParameters: document.getElementById("training-parameters"),
  trainingStartButton: document.getElementById("training-start-button"),
  trainingStatus: document.getElementById("training-status"),
  trainingJobs: document.getElementById("training-jobs"),
  artifactTemplate: document.getElementById("artifact-template"),
};

function setStatus(target, message, tone = "") {
  target.textContent = message || "";
  target.className = "status-line";
  if (tone) {
    target.classList.add(tone);
  }
}

function switchTab(tabName) {
  elements.tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  elements.tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
}

function jsonBlock(payload) {
  const pre = document.createElement("pre");
  pre.className = "json-view";
  pre.textContent = JSON.stringify(payload, null, 2);
  return pre;
}

function textBlock(text, className = "text-view") {
  const pre = document.createElement("pre");
  pre.className = className;
  pre.textContent = text;
  return pre;
}

function renderArtifacts(artifacts) {
  if (!artifacts || artifacts.length === 0) {
    return null;
  }
  const grid = document.createElement("div");
  grid.className = "artifact-grid";
  artifacts.forEach((artifact) => {
    const fragment = elements.artifactTemplate.content.cloneNode(true);
    fragment.querySelector(".artifact-label").textContent = artifact.label;
    const link = fragment.querySelector(".artifact-link");
    link.href = artifact.url;
    const body = fragment.querySelector(".artifact-body");
    if (artifact.kind === "image") {
      const image = document.createElement("img");
      image.src = artifact.url;
      image.alt = artifact.label;
      body.appendChild(image);
    } else {
      body.appendChild(textBlock(artifact.relative_path));
    }
    grid.appendChild(fragment);
  });
  return grid;
}

function renderPayload(container, payload) {
  container.innerHTML = "";
  if (!payload) {
    container.appendChild(textBlock("No data"));
    return;
  }
  if (payload.summary) {
    const summary = document.createElement("p");
    summary.textContent = payload.summary;
    container.appendChild(summary);
  }
  const artifacts = renderArtifacts(payload.artifacts);
  if (artifacts) {
    container.appendChild(artifacts);
  }
  if (payload.text) {
    container.appendChild(textBlock(payload.text));
  }
  if (payload.data) {
    container.appendChild(jsonBlock(payload.data));
  }
}

function renderStageResults(result) {
  elements.ocrResults.innerHTML = "";
  if (!result || !result.stage_results || result.stage_results.length === 0) {
    return;
  }

  result.stage_results.forEach((stageResult) => {
    const card = document.createElement("article");
    card.className = "stage-card";
    card.innerHTML = `
      <div class="stage-card-header">
        <h3>${stageResult.title}</h3>
        <span>${stageResult.stage}</span>
      </div>
      <div class="stage-card-layout">
        <section class="payload-panel">
          <h4>Input</h4>
          <div class="payload-body input-payload"></div>
        </section>
        <section class="payload-panel">
          <h4>Output</h4>
          <div class="payload-body output-payload"></div>
        </section>
      </div>
    `;
    renderPayload(card.querySelector(".input-payload"), stageResult.input);
    renderPayload(card.querySelector(".output-payload"), stageResult.output);
    elements.ocrResults.appendChild(card);
  });
}

function renderStructuredResult(result) {
  elements.ocrStructuredResult.innerHTML = "";
  if (!result) {
    return;
  }

  const card = document.createElement("article");
  card.className = "stage-card";
  card.innerHTML = `
    <div class="stage-card-header">
      <h3>${result.record_type || "unknown"}</h3>
      <span>${result.document_metadata?.human_review_required ? "review required" : "ready"}</span>
    </div>
  `;
  card.appendChild(jsonBlock(result));
  elements.ocrStructuredResult.appendChild(card);
}

function getSelectedStages() {
  return Array.from(document.querySelectorAll("input[name='ocr-stage']:checked")).map((input) => input.value);
}

function getStageGapMessage(selectedStages) {
  if (!selectedStages || selectedStages.length === 0) {
    return "Choose at least one stage to configure the manual input.";
  }

  const indexes = selectedStages
    .map((stageName) => state.ocrConfig.ocr_stages.findIndex((item) => item.name === stageName))
    .filter((index) => index >= 0);

  for (let index = 1; index < indexes.length; index += 1) {
    if (indexes[index] !== indexes[index - 1] + 1) {
      return "Selected stages must form one continuous pipeline slice.";
    }
  }
  return null;
}

function updateOcrInputControls() {
  state.selectedStages = getSelectedStages();
  const gapMessage = getStageGapMessage(state.selectedStages);
  const firstStage = state.selectedStages[0];
  const stageConfig = state.ocrConfig.ocr_stages.find((item) => item.name === firstStage);

  elements.ocrFileRow.classList.add("hidden");
  elements.ocrTextRow.classList.add("hidden");
  elements.ocrManualLabelRow.classList.add("hidden");
  elements.ocrFiles.multiple = false;
  elements.ocrRunButton.disabled = false;
  elements.ocrProcessButton.disabled = false;

  if (gapMessage) {
    elements.ocrInputHelp.textContent = gapMessage;
    elements.ocrRunButton.disabled = true;
    return;
  }

  if (stageConfig.input_kind === "image") {
    elements.ocrFileRow.classList.remove("hidden");
    elements.ocrFiles.multiple = false;
    elements.ocrInputHelp.textContent = `Upload one image for ${stageConfig.title}.`;
  } else if (stageConfig.input_kind === "segments") {
    elements.ocrFileRow.classList.remove("hidden");
    elements.ocrFiles.multiple = true;
    elements.ocrInputHelp.textContent = `Upload one or more segment images for ${stageConfig.title}.`;
    if (firstStage === "ocr") {
      elements.ocrManualLabelRow.classList.remove("hidden");
    }
    elements.ocrProcessButton.disabled = true;
  } else if (stageConfig.input_kind === "text") {
    elements.ocrTextRow.classList.remove("hidden");
    elements.ocrInputHelp.textContent = "Paste the input text for NER.";
    elements.ocrProcessButton.disabled = true;
  }
}

function renderOcrStageControls() {
  elements.ocrStageList.innerHTML = "";
  state.ocrConfig.ocr_stages.forEach((stage, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "stage-checkbox";
    wrapper.innerHTML = `
      <label>
        <input type="checkbox" name="ocr-stage" value="${stage.name}" ${index < state.ocrConfig.ocr_stages.length ? "checked" : ""}>
        ${stage.title}
      </label>
      <div class="stage-description">${stage.description}</div>
    `;
    elements.ocrStageList.appendChild(wrapper);
  });

  document.querySelectorAll("input[name='ocr-stage']").forEach((input) => {
    input.addEventListener("change", updateOcrInputControls);
  });

  elements.ocrHandwrittenModel.innerHTML = "";
  state.ocrConfig.handwritten_models.forEach((modelName) => {
    const option = document.createElement("option");
    option.value = modelName;
    option.textContent = modelName;
    elements.ocrHandwrittenModel.appendChild(option);
  });
  updateOcrInputControls();
}

function trainingParameterValue(field, parameter) {
  if (parameter.type === "int") {
    if (field.value === "") return null;
    return Number.parseInt(field.value, 10);
  }
  if (parameter.type === "float") {
    if (field.value === "") return null;
    return Number.parseFloat(field.value);
  }
  return field.value;
}

function renderTrainingModelDetails() {
  const selectedKey = elements.trainingModelSelect.value;
  state.selectedTrainingModel = state.trainingModels.find((model) => model.key === selectedKey) || null;
  elements.trainingParameters.innerHTML = "";

  if (!state.selectedTrainingModel) {
    elements.trainingModelStatus.textContent = "No training model selected.";
    elements.trainingModelStatus.className = "status-chip";
    elements.trainingStartButton.disabled = true;
    return;
  }

  elements.trainingModelStatus.className = `status-chip ${state.selectedTrainingModel.available ? "success" : "danger"}`;
  elements.trainingModelStatus.textContent = state.selectedTrainingModel.available
    ? `Dataset: ${state.selectedTrainingModel.dataset_path}`
    : state.selectedTrainingModel.reason || "Unavailable";

  state.selectedTrainingModel.parameters.forEach((parameter) => {
    const field = document.createElement("div");
    field.className = "parameter-field";
    field.innerHTML = `
      <label for="param-${parameter.name}">${parameter.name}</label>
      <input
        id="param-${parameter.name}"
        data-parameter="${parameter.name}"
        type="${parameter.type === "float" || parameter.type === "int" ? "number" : "text"}"
        step="${parameter.type === "float" ? "any" : "1"}"
        value="${parameter.default ?? ""}"
      >
    `;
    elements.trainingParameters.appendChild(field);
  });

  elements.trainingStartButton.disabled = !state.selectedTrainingModel.available;
}

function renderTrainingControls() {
  elements.trainingModelSelect.innerHTML = "";
  state.trainingModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.key;
    option.textContent = `${model.title}${model.available ? "" : " (unavailable)"}`;
    elements.trainingModelSelect.appendChild(option);
  });
  renderTrainingModelDetails();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail || response.statusText);
  }
  return response.json();
}

async function runStructuredExtraction() {
  const files = Array.from(elements.ocrFiles.files);
  if (files.length !== 1) {
    setStatus(elements.ocrStatus, "Structured extraction expects exactly one page image.", "danger");
    return;
  }

  const formData = new FormData();
  formData.append("handwritten_model", elements.ocrHandwrittenModel.value);
  formData.append("include_debug", "false");
  formData.append("files", files[0]);

  setStatus(elements.ocrStatus, "Running full OCR extraction...");
  elements.ocrStructuredResult.innerHTML = "";
  try {
    const result = await fetchJson("/api/ocr/process", {
      method: "POST",
      body: formData,
    });
    renderStructuredResult(result);
    setStatus(
      elements.ocrStatus,
      `Structured OCR run ${result.document_metadata?.run_id || ""} completed.`.trim()
    );
  } catch (error) {
    setStatus(elements.ocrStatus, error.message, "danger");
  }
}

async function runOcrStages() {
  const selectedStages = getSelectedStages();
  if (selectedStages.length === 0) {
    setStatus(elements.ocrStatus, "Select at least one OCR stage.", "danger");
    return;
  }
  const gapMessage = getStageGapMessage(selectedStages);
  if (gapMessage) {
    setStatus(elements.ocrStatus, gapMessage, "danger");
    return;
  }

  const firstStage = selectedStages[0];
  const formData = new FormData();
  formData.append("stages", JSON.stringify(selectedStages));
  formData.append("handwritten_model", elements.ocrHandwrittenModel.value);
  if (firstStage === "ner") {
    formData.append("text_input", elements.ocrText.value);
  } else {
    Array.from(elements.ocrFiles.files).forEach((file) => formData.append("files", file));
    if (firstStage === "ocr") {
      formData.append("manual_segment_label", elements.ocrManualLabel.value);
    }
  }

  setStatus(elements.ocrStatus, "Running selected OCR stages...");
  elements.ocrResults.innerHTML = "";
  try {
    const result = await fetchJson("/api/ocr/run", {
      method: "POST",
      body: formData,
    });
    renderStageResults(result);
    setStatus(elements.ocrStatus, `Debug run ${result.run_id} completed.`);
  } catch (error) {
    setStatus(elements.ocrStatus, error.message, "danger");
  }
}

function renderJobCard(job) {
  const article = document.createElement("article");
  article.className = "job-card";
  const statusTone = job.status === "completed" ? "success" : job.status === "failed" ? "danger" : "warning";
  article.innerHTML = `
    <div class="job-header">
      <div>
        <strong>${job.model_title}</strong>
        <span>${job.job_id}</span>
      </div>
      <span class="status-chip ${statusTone}">${job.status}</span>
    </div>
    <div class="job-meta">
      <div>Dataset: ${job.dataset_path || "n/a"}</div>
      <div>Output: ${job.output_dir || "n/a"}</div>
    </div>
    <div class="job-links">
      <a href="/api/training/jobs/${job.job_id}/log" target="_blank" rel="noreferrer">Log JSON</a>
    </div>
  `;
  if (job.log_tail) {
    article.appendChild(textBlock(job.log_tail, "log-view"));
  }
  return article;
}

async function refreshTrainingJobs() {
  try {
    const jobs = await fetchJson("/api/training/jobs");
    elements.trainingJobs.innerHTML = "";
    if (jobs.length === 0) {
      elements.trainingJobs.appendChild(textBlock("No training jobs yet."));
      return;
    }
    jobs.forEach((job) => elements.trainingJobs.appendChild(renderJobCard(job)));
  } catch (error) {
    elements.trainingJobs.innerHTML = "";
    elements.trainingJobs.appendChild(textBlock(error.message, "log-view"));
  }
}

async function startTrainingJob() {
  if (!state.selectedTrainingModel || !state.selectedTrainingModel.available) {
    setStatus(elements.trainingStatus, "Selected model is unavailable.", "danger");
    return;
  }

  const parameters = {};
  state.selectedTrainingModel.parameters.forEach((parameter) => {
    const field = document.getElementById(`param-${parameter.name}`);
    parameters[parameter.name] = trainingParameterValue(field, parameter);
  });

  setStatus(elements.trainingStatus, "Starting training job...");
  try {
    const job = await fetchJson("/api/training/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_key: state.selectedTrainingModel.key,
        parameters,
      }),
    });
    setStatus(elements.trainingStatus, `Training job ${job.job_id} started.`);
    await refreshTrainingJobs();
  } catch (error) {
    setStatus(elements.trainingStatus, error.message, "danger");
  }
}

async function bootstrap() {
  const [ocrConfig, trainingModels] = await Promise.all([
    fetchJson("/api/ocr/config"),
    fetchJson("/api/training/models"),
  ]);

  state.ocrConfig = ocrConfig;
  state.trainingModels = trainingModels;

  renderOcrStageControls();
  renderTrainingControls();
  await refreshTrainingJobs();

  elements.tabButtons.forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  elements.ocrProcessButton.addEventListener("click", runStructuredExtraction);
  elements.ocrRunButton.addEventListener("click", runOcrStages);
  elements.trainingModelSelect.addEventListener("change", renderTrainingModelDetails);
  elements.trainingStartButton.addEventListener("click", startTrainingJob);

  if (state.jobsPollHandle) {
    clearInterval(state.jobsPollHandle);
  }
  state.jobsPollHandle = window.setInterval(refreshTrainingJobs, 5000);
}

bootstrap().catch((error) => {
  document.body.innerHTML = `<pre class="log-view">${error.message}</pre>`;
});
