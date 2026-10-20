// static/script.js - Interactive Logic for Gujarati Spell Checker

document.addEventListener("DOMContentLoaded", () => {
  const textInput = document.getElementById("text-input");
  const checkBtn = document.getElementById("check-btn");
  const clearBtn = document.getElementById("clear-btn");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");

  const resultSection = document.getElementById("result-section");
  const verdictBanner = document.getElementById("verdict-banner");
  const summaryEntered = document.getElementById("summary-entered");
  const summaryCorrected = document.getElementById("summary-corrected");
  const copyResultBtn = document.getElementById("copy-result-btn");
  const suggestionsBox = document.getElementById("suggestions-box");
  const suggestionsBar = document.getElementById("suggestions-bar");
  const tokensList = document.getElementById("tokens-list");

  const toggleDetailsBtn = document.getElementById("toggle-details-btn");
  const detailsPanel = document.getElementById("processing-details-panel");
  const toggleIcon = toggleDetailsBtn ? toggleDetailsBtn.querySelector(".toggle-icon") : null;
  const toggleText = toggleDetailsBtn ? toggleDetailsBtn.querySelector(".toggle-text") : null;

  // Pipeline elements
  const pipeUnicode = document.getElementById("pipe-unicode");
  const pipeWhitespace = document.getElementById("pipe-whitespace");
  const pipePunctuation = document.getElementById("pipe-punctuation");
  const pipeTokens = document.getElementById("pipe-tokens");
  const pipeVocabTable = document.getElementById("pipe-vocab-table");
  const pipeCandidatesContainer = document.getElementById("pipe-candidates-container");
  const pipeNgramContainer = document.getElementById("pipe-ngram-container");

  // State variable to store original check result for deterministic replacement
  let activeCheckResult = null;

  // Copy button handler
  if (copyResultBtn) {
    copyResultBtn.addEventListener("click", () => {
      const textToCopy = summaryCorrected.textContent.trim();
      if (!textToCopy) return;

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(textToCopy).then(() => {
          triggerCopiedFeedback();
        }).catch(() => {
          fallbackCopyText(textToCopy);
        });
      } else {
        fallbackCopyText(textToCopy);
      }
    });
  }

  function fallbackCopyText(text) {
    const tempTextarea = document.createElement("textarea");
    tempTextarea.value = text;
    document.body.appendChild(tempTextarea);
    tempTextarea.select();
    try {
      document.execCommand("copy");
      triggerCopiedFeedback();
    } catch (err) {
      console.error("Fallback copy failed", err);
    }
    document.body.removeChild(tempTextarea);
  }

  function triggerCopiedFeedback() {
    copyResultBtn.classList.add("copied");
    const copyLabel = copyResultBtn.querySelector(".copy-text");
    if (copyLabel) copyLabel.textContent = "Copied!";
    setTimeout(() => {
      copyResultBtn.classList.remove("copied");
      if (copyLabel) copyLabel.textContent = "Copy";
    }, 2000);
  }

  // Check button click
  checkBtn.addEventListener("click", () => {
    const text = textInput.value.trim();
    if (!text) {
      alert("Please enter a Gujarati word or sentence.");
      textInput.focus();
      return;
    }
    runSpellCheck(text);
  });

  // Enter key in textarea (Enter without Shift)
  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      checkBtn.click();
    }
  });

  // Clear button click
  clearBtn.addEventListener("click", () => {
    textInput.value = "";
    resultSection.style.display = "none";
    if (suggestionsBox) suggestionsBox.style.display = "none";
    if (detailsPanel) detailsPanel.classList.remove("open");
    if (toggleIcon) toggleIcon.classList.remove("open");
    if (toggleText) toggleText.textContent = "View Complete Processing Pipeline";
    activeCheckResult = null;
    textInput.focus();
  });

  // Toggle details button click with smooth accordion animation
  if (toggleDetailsBtn) {
    toggleDetailsBtn.addEventListener("click", () => {
      const isOpen = detailsPanel.classList.contains("open");
      if (isOpen) {
        detailsPanel.classList.remove("open");
        if (toggleIcon) toggleIcon.classList.remove("open");
        if (toggleText) toggleText.textContent = "View Complete Processing Pipeline";
      } else {
        detailsPanel.classList.add("open");
        if (toggleIcon) toggleIcon.classList.add("open");
        if (toggleText) toggleText.textContent = "Hide Processing Pipeline";
      }
    });
  }

  // Execute spell check API request
  async function runSpellCheck(text) {
    setLoading(true);

    try {
      const response = await fetch("/check", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        alert(data.error || "An error occurred during spell checking.");
        setLoading(false);
        return;
      }

      displayResults(data);
    } catch (err) {
      console.error(err);
      alert("Network error: Unable to contact spell-checker server.");
    } finally {
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    checkBtn.disabled = isLoading;
    if (isLoading) {
      btnText.textContent = "Checking...";
      btnSpinner.style.display = "inline-block";
    } else {
      btnText.textContent = "Check Spelling";
      btnSpinner.style.display = "none";
    }
  }

  function displayResults(data) {
    activeCheckResult = data;
    resultSection.style.display = "block";

    // Verdict Banner: clean, no technical explanations
    if (!data.has_error) {
      verdictBanner.className = "verdict-banner correct";
      verdictBanner.innerHTML = "<span>✓</span> <div><strong>સાચી જોડણી (Correct Spelling)</strong></div>";
    } else {
      verdictBanner.className = "verdict-banner error";
      verdictBanner.innerHTML = "<span>⚠</span> <div><strong>સંભવિત જોડણી ભૂલ (Possible Spelling Error)</strong></div>";
    }

    // Summary Card
    summaryEntered.textContent = data.original_text;
    summaryCorrected.textContent = data.corrected_text;

    // Google Keyboard Style Suggestions Bar
    if (data.has_error && data.first_error && data.first_error.top_candidates && data.first_error.top_candidates.length > 0) {
      suggestionsBox.style.display = "block";
      suggestionsBar.innerHTML = "";

      const misspelledWord = data.first_error.word;

      data.first_error.top_candidates.forEach((cand, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `suggestion-chip ${idx === 0 ? "top-suggestion" : ""}`;
        btn.textContent = cand.word;
        btn.title = `Edit Distance: ${cand.levenshtein_distance}`;

        // Clicking a suggestion chip replaces the word cleanly without punctuation accumulation
        btn.addEventListener("click", () => {
          applySuggestion(cand.word);
          // Highlight clicked chip
          document.querySelectorAll(".suggestion-chip").forEach(c => c.classList.remove("top-suggestion"));
          btn.classList.add("top-suggestion");
        });

        suggestionsBar.appendChild(btn);
      });
    } else {
      if (suggestionsBox) suggestionsBox.style.display = "none";
    }

    // Token Pills
    tokensList.innerHTML = "";
    (data.words_analysis || []).forEach((item) => {
      const pill = document.createElement("div");
      pill.className = `token-pill ${item.is_correct ? "is-correct" : "is-error"}`;
      pill.innerHTML = `
        <span>${escapeHtml(item.word)}</span>
        <span class="pill-status">${item.is_correct ? "✓ Correct" : "⚠ Error"}</span>
      `;
      tokensList.appendChild(pill);
    });

    // Populate Visible Pipeline Details
    const p = data.preprocessing_details || {};
    pipeUnicode.textContent = p.unicode_normalized || data.original_text;
    pipeWhitespace.textContent = p.whitespace_normalized || data.original_text;
    pipePunctuation.textContent = p.punctuation_handled || data.original_text;
    pipeTokens.textContent = JSON.stringify(p.tokens || data.tokens);

    // Vocabulary Table
    pipeVocabTable.innerHTML = "";
    (data.words_analysis || []).forEach((item, index) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${index + 1}</td>
        <td><strong>${escapeHtml(item.word)}</strong></td>
        <td>
          <span class="status-tag ${item.in_vocabulary ? "found" : "not-found"}">
            ${item.in_vocabulary ? "✓ Found" : "✗ Not Found"}
          </span>
        </td>
        <td>${item.in_vocabulary ? "Correct word in vocabulary" : "Not found in vocabulary"}</td>
      `;
      pipeVocabTable.appendChild(tr);
    });

    // Step 7: Candidate Generation & Levenshtein
    pipeCandidatesContainer.innerHTML = "";
    const errorItems = (data.words_analysis || []).filter((item) => !item.is_correct);

    if (errorItems.length === 0) {
      pipeCandidatesContainer.innerHTML = '<p class="candidate-detail">No misspelled words found; candidate generation not required.</p>';
    } else {
      errorItems.forEach((errItem) => {
        const wrap = document.createElement("div");
        wrap.className = "candidate-group";
        wrap.innerHTML = `<h5 style="margin-top:0.75rem; font-family:var(--font-gujarati); color:#93c5fd">Misspelled Word: '${escapeHtml(errItem.word)}'</h5>`;

        if (!errItem.top_candidates || errItem.top_candidates.length === 0) {
          wrap.innerHTML += '<p class="candidate-detail">No close candidate found in vocabulary.</p>';
        } else {
          errItem.top_candidates.forEach((cand) => {
            const card = document.createElement("div");
            card.className = "candidate-item";
            card.innerHTML = `
              <div class="candidate-header">
                <span class="candidate-word">${escapeHtml(cand.word)}</span>
                <div class="candidate-metrics">
                  <span class="metric-badge">Levenshtein Dist: <strong>${cand.levenshtein_distance}</strong></span>
                  <span class="metric-badge">Context Score: <strong>${cand.context_score}</strong></span>
                </div>
              </div>
              <div class="candidate-detail">${escapeHtml(cand.explanation)}</div>
            `;
            wrap.appendChild(card);
          });
        }
        pipeCandidatesContainer.appendChild(wrap);
      });
    }

    // Step 8: N-gram Context Ranking Details
    pipeNgramContainer.innerHTML = "";
    if (errorItems.length === 0) {
      pipeNgramContainer.innerHTML = '<p class="candidate-detail">All tokens match vocabulary. Contextual ranking unnecessary.</p>';
    } else {
      errorItems.forEach((errItem) => {
        const div = document.createElement("div");
        div.className = "candidate-detail";
        div.style.marginTop = "0.5rem";
        div.innerHTML = `
          <strong>Decision for '${escapeHtml(errItem.word)}':</strong> 
          Selected candidate: <strong>'${escapeHtml(errItem.suggested_correction || "None")}'</strong>
          <br>
          <em>Levenshtein Filter + N-gram Scoring: ${escapeHtml(errItem.context_info)}</em>
        `;
        pipeNgramContainer.appendChild(div);
      });
    }
  }

  /**
   * Deterministic suggestion replacement:
   * Always derives from activeCheckResult.original_text, so clicking multiple
   * suggestions sequentially switches cleanly without accumulating matras or punctuation.
   */
  function applySuggestion(replacementWord) {
    if (!activeCheckResult || !activeCheckResult.first_error) return;

    const originalMisspelled = activeCheckResult.first_error.word;

    // 1. Update Suggested Result display
    const correctedTokens = (activeCheckResult.words_analysis || []).map((item) => {
      if (!item.is_correct && item.word === originalMisspelled) {
        return replacementWord;
      }
      return item.is_correct ? item.word : (item.suggested_correction || item.word);
    });
    summaryCorrected.textContent = correctedTokens.join(" ");

    // 2. Update Input Textarea by replacing the original misspelled word in activeCheckResult.original_text
    const baseText = activeCheckResult.original_text;
    const escaped = originalMisspelled.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const wordPattern = new RegExp(`(^|\\s)${escaped}(?=[\\s.,!?:;।॥"'\\]\\)\\}\\-]|\$)`, "gu");

    if (wordPattern.test(baseText)) {
      textInput.value = baseText.replace(wordPattern, `$1${replacementWord}`);
    } else {
      textInput.value = baseText.replace(originalMisspelled, replacementWord);
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
