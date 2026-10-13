/**
 * EduPredict – Student Academic Performance Prediction System
 * Frontend Controller: static/script.js
 * Light-Purple Theme with Circular Score Gauge & Intelligent Academic Insights
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const form = document.getElementById("predictionForm");
  const submitBtn = document.getElementById("submitBtn");
  const submitText = document.getElementById("submitText");
  const spinner = document.getElementById("submitSpinner");
  const resetBtn = document.getElementById("resetBtn");

  const alertBanner = document.getElementById("alertBanner");
  const alertText = document.getElementById("alertText");

  const stateEmpty = document.getElementById("stateEmpty");
  const stateLoading = document.getElementById("stateLoading");
  const stateResult = document.getElementById("stateResult");

  const scoreValueEl = document.getElementById("scoreValue");
  const scoreRingCircle = document.getElementById("scoreRingCircle");
  const scoreTierBadge = document.getElementById("scoreTierBadge");
  const insightTextEl = document.getElementById("insightText");

  // Metric Display Elements
  const metricHours = document.getElementById("metricHours");
  const metricScores = document.getElementById("metricScores");
  const metricExtra = document.getElementById("metricExtra");
  const metricSleep = document.getElementById("metricSleep");
  const metricPapers = document.getElementById("metricPapers");

  // Health Status Elements
  const statusPill = document.getElementById("healthStatusPill");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const livePill = document.getElementById("livePill");

  const CIRCLE_CIRCUMFERENCE = 415; // 2 * PI * 66

  // ============================================================================
  // 1. Health Status Polling & Model Verification
  // ============================================================================
  async function checkHealthStatus() {
    try {
      const response = await fetch("/health");
      if (response.ok) {
        const data = await response.json();
        if (data.model_loaded) {
          statusDot.className = "status-dot active";
          statusText.textContent = "ML Model Active";
          statusPill.title = `Model loaded from: ${data.model_path || "disk"}`;
          if (livePill) livePill.textContent = "● Live Model Ready";
        } else {
          statusDot.className = "status-dot warning";
          statusText.textContent = "Model Not Trained";
          statusPill.title = "Backend running, but model not trained yet. Run train_model.py.";
          if (livePill) livePill.textContent = "⚠️ Model Unloaded";
        }
      } else {
        throw new Error("HTTP " + response.status);
      }
    } catch (err) {
      statusDot.className = "status-dot";
      statusText.textContent = "Server Offline";
      if (livePill) livePill.textContent = "✕ Disconnected";
    }
  }

  checkHealthStatus();

  // ============================================================================
  // 2. Preset Profile Handlers (High Achiever, Balanced, Needs Boost)
  // ============================================================================
  const presetProfiles = {
    achiever: {
      hours_studied: 8,
      previous_scores: 92,
      extracurricular_activities: "Yes",
      sleep_hours: 8,
      sample_question_papers_practiced: 6
    },
    balanced: {
      hours_studied: 5.5,
      previous_scores: 74,
      extracurricular_activities: "Yes",
      sleep_hours: 7.5,
      sample_question_papers_practiced: 4
    },
    support: {
      hours_studied: 2.5,
      previous_scores: 48,
      extracurricular_activities: "No",
      sleep_hours: 5.5,
      sample_question_papers_practiced: 1
    }
  };

  document.querySelectorAll(".btn-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      const profile = presetProfiles[presetKey];
      if (profile) {
        populateForm(profile);
        clearAlert();
        clearValidationErrors();
      }
    });
  });

  function populateForm(data) {
    document.getElementById("hours_studied").value = data.hours_studied;
    document.getElementById("previous_scores").value = data.previous_scores;
    document.getElementById("sleep_hours").value = data.sleep_hours;
    document.getElementById("sample_question_papers_practiced").value = data.sample_question_papers_practiced;

    const radio = document.querySelector(
      `input[name="extracurricular_activities"][value="${data.extracurricular_activities}"]`
    );
    if (radio) radio.checked = true;
  }

  // ============================================================================
  // 3. Client Validation
  // ============================================================================
  function showAlert(message, type = "error") {
    alertBanner.className = `alert-banner ${type}`;
    alertText.textContent = message;
    alertBanner.style.display = "flex";
  }

  function clearAlert() {
    alertBanner.style.display = "none";
    alertBanner.className = "alert-banner";
  }

  function clearValidationErrors() {
    document.querySelectorAll(".form-input").forEach((el) => {
      el.classList.remove("invalid");
    });
  }

  function validateInputs() {
    clearValidationErrors();
    let isValid = true;
    let firstErrorMsg = "";

    const hoursInput = document.getElementById("hours_studied");
    const hours = parseFloat(hoursInput.value);
    if (isNaN(hours) || hours < 0 || hours > 24) {
      hoursInput.classList.add("invalid");
      isValid = false;
      firstErrorMsg = firstErrorMsg || "Daily study hours must be a number between 0 and 24.";
    }

    const scoresInput = document.getElementById("previous_scores");
    const scores = parseFloat(scoresInput.value);
    if (isNaN(scores) || scores < 0 || scores > 100) {
      scoresInput.classList.add("invalid");
      isValid = false;
      firstErrorMsg = firstErrorMsg || "Previous scores must be a percentage between 0 and 100.";
    }

    const extraInput = document.querySelector('input[name="extracurricular_activities"]:checked');
    if (!extraInput) {
      isValid = false;
      firstErrorMsg = firstErrorMsg || "Please select Extracurricular Activities participation.";
    }

    const sleepInput = document.getElementById("sleep_hours");
    const sleep = parseFloat(sleepInput.value);
    if (isNaN(sleep) || sleep < 0 || sleep > 24) {
      sleepInput.classList.add("invalid");
      isValid = false;
      firstErrorMsg = firstErrorMsg || "Daily sleep hours must be between 0 and 24.";
    }

    const papersInput = document.getElementById("sample_question_papers_practiced");
    const papers = parseInt(papersInput.value, 10);
    if (isNaN(papers) || papers < 0) {
      papersInput.classList.add("invalid");
      isValid = false;
      firstErrorMsg = firstErrorMsg || "Sample question papers must be 0 or greater.";
    }

    if (!isValid) {
      showAlert(firstErrorMsg, "error");
    }

    return isValid;
  }

  // ============================================================================
  // 4. Form Submission & Fetch API
  // ============================================================================
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearAlert();

    if (!validateInputs()) {
      return;
    }

    const payload = {
      hours_studied: parseFloat(document.getElementById("hours_studied").value),
      previous_scores: parseFloat(document.getElementById("previous_scores").value),
      extracurricular_activities: document.querySelector('input[name="extracurricular_activities"]:checked').value,
      sleep_hours: parseFloat(document.getElementById("sleep_hours").value),
      sample_question_papers_practiced: parseInt(document.getElementById("sample_question_papers_practiced").value, 10)
    };

    setLoadingState(true);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (response.ok && data.status === "success") {
        renderPredictionSuccess(data.prediction, payload);
      } else if (response.status === 503) {
        showEmptyState();
        showAlert(
          data.error || "The ML model pipeline is not yet trained. Run 'python train_model.py' first.",
          "warning"
        );
      } else {
        showEmptyState();
        showAlert(data.error || "Prediction request could not be completed.", "error");
      }
    } catch (err) {
      showEmptyState();
      showAlert("Network error: Unable to connect to prediction server.", "error");
    } finally {
      setLoadingState(false);
    }
  });

  // Reset Button
  resetBtn.addEventListener("click", () => {
    form.reset();
    clearAlert();
    clearValidationErrors();
    showEmptyState();
  });

  // UI State Switchers
  function setLoadingState(isLoading) {
    if (isLoading) {
      submitBtn.disabled = true;
      spinner.style.display = "inline-block";
      submitText.textContent = "Analyzing...";

      stateEmpty.style.display = "none";
      stateResult.style.display = "none";
      stateLoading.style.display = "flex";
    } else {
      submitBtn.disabled = false;
      spinner.style.display = "none";
      submitText.textContent = "Predict Performance Index";
    }
  }

  function showEmptyState() {
    stateLoading.style.display = "none";
    stateResult.style.display = "none";
    stateEmpty.style.display = "flex";
    if (scoreRingCircle) {
      scoreRingCircle.style.strokeDashoffset = CIRCLE_CIRCUMFERENCE;
    }
  }

  // ============================================================================
  // 5. Intelligent Guidance & Score Gauge Rendering
  // ============================================================================
  function renderPredictionSuccess(predictedScore, inputs) {
    stateEmpty.style.display = "none";
    stateLoading.style.display = "none";
    stateResult.style.display = "block";

    // Animate circular progress ring and number counter
    animateScoreGauge(predictedScore);

    // Determine Tier Badge
    let tierText = "";
    let tierClass = "";

    if (predictedScore >= 85) {
      tierText = "Outstanding Potential";
      tierClass = "tier-high";
    } else if (predictedScore >= 70) {
      tierText = "High Proficiency";
      tierClass = "tier-mid";
    } else if (predictedScore >= 50) {
      tierText = "Moderate Growth";
      tierClass = "tier-mid";
    } else {
      tierText = "Academic Support Advised";
      tierClass = "tier-low";
    }

    scoreTierBadge.textContent = tierText;
    scoreTierBadge.className = `score-badge ${tierClass}`;

    // Fill metrics tiles
    metricHours.textContent = `${inputs.hours_studied} hrs/day`;
    metricScores.textContent = `${inputs.previous_scores}%`;
    metricExtra.textContent = inputs.extracurricular_activities === "Yes" ? "Active" : "Inactive";
    metricSleep.textContent = `${inputs.sleep_hours} hrs/day`;
    metricPapers.textContent = `${inputs.sample_question_papers_practiced} papers`;

    // Generate Personalized Academic Guidance based on BOTH predicted score and input metrics
    const guidance = generatePersonalizedGuidance(predictedScore, inputs);
    insightTextEl.innerHTML = guidance;
  }

  function generatePersonalizedGuidance(score, inputs) {
    const tips = [];

    // Study Hours advice
    if (inputs.hours_studied < 4) {
      tips.push("Increasing daily study sessions to 5–6 hours with focused intervals (Pomodoro technique) could boost your index by 8–12 points.");
    } else if (inputs.hours_studied >= 7) {
      tips.push("Excellent study dedication! Maintain consistency without risking mental burnout.");
    }

    // Previous Scores advice
    if (inputs.previous_scores < 60) {
      tips.push("Revisit fundamental concepts from past coursework with a mentor to solidify foundational knowledge.");
    } else if (inputs.previous_scores >= 85) {
      tips.push("Your strong academic foundation provides high confidence for advanced exam challenges.");
    }

    // Sleep advice
    if (inputs.sleep_hours < 6.5) {
      tips.push("⚠️ Rest Alert: You sleep under 6.5 hours. Neuroscience research shows 7–8 hours of regular sleep significantly enhances memory retention and problem solving.");
    } else if (inputs.sleep_hours >= 7 && inputs.sleep_hours <= 9) {
      tips.push("Great rest cycle! Optimal sleep directly supports sustained analytical endurance during exams.");
    }

    // Practice Papers advice
    if (inputs.sample_question_papers_practiced < 3) {
      tips.push("Practicing 3 to 5 additional timed mock papers will improve test-taking speed and reduce exam anxiety.");
    } else {
      tips.push("Great sample paper preparation! Regular mock test practice keeps exam technique sharp.");
    }

    // Extracurriculars
    if (inputs.extracurricular_activities === "No") {
      tips.push("Consider participating in light physical sports or school clubs to reduce stress and boost cognitive balance.");
    }

    // Combine top recommendations
    return `<strong>Performance Assessment:</strong> The trained model forecasts an index of <strong>${score} / 100</strong>.<br><br>• ${tips.slice(0, 3).join("<br><br>• ")}`;
  }

  function animateScoreGauge(targetScore) {
    const duration = 1000;
    const startTime = performance.now();
    const clampedScore = Math.max(0, Math.min(100, targetScore));

    // Calculate circular stroke offset (0 score = 415, 100 score = 0)
    const targetOffset = CIRCLE_CIRCUMFERENCE - (clampedScore / 100) * CIRCLE_CIRCUMFERENCE;
    if (scoreRingCircle) {
      scoreRingCircle.style.strokeDashoffset = targetOffset;
    }

    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3); // cubic ease-out
      const currentVal = (targetScore * ease).toFixed(1);

      scoreValueEl.textContent = currentVal;

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        scoreValueEl.textContent = targetScore.toFixed(2);
      }
    }

    requestAnimationFrame(step);
  }
});
