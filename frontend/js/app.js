"use strict";


/* ==========================================================
   API CONFIGURATION
========================================================== */

const API_BASE_URL = "http://127.0.0.1:8000";

const PREDICT_ENDPOINT =
    `${API_BASE_URL}/api/predict`;

const HEALTH_ENDPOINT =
    `${API_BASE_URL}/api/health`;


/* ==========================================================
   APPLICATION SETTINGS
========================================================== */

const MIN_WORDS = 20;


/* ==========================================================
   DOM ELEMENTS
========================================================== */

const headlineInput =
    document.getElementById("headline");

const articleInput =
    document.getElementById("article");

const wordCountElement =
    document.getElementById("wordCount");


const analyzeButton =
    document.getElementById(
        "analyzeButton"
    );

const buttonText =
    document.getElementById(
        "buttonText"
    );

const buttonSpinner =
    document.getElementById(
        "buttonSpinner"
    );


const errorMessage =
    document.getElementById(
        "errorMessage"
    );


const loadingSection =
    document.getElementById(
        "loadingSection"
    );


const resultSection =
    document.getElementById(
        "resultSection"
    );


const finalResultCard =
    document.getElementById(
        "finalResultCard"
    );

const finalPrediction =
    document.getElementById(
        "finalPrediction"
    );

const finalBadge =
    document.getElementById(
        "finalBadge"
    );

const finalConfidence =
    document.getElementById(
        "finalConfidence"
    );

const selectedModel =
    document.getElementById(
        "selectedModel"
    );

const modelAgreement =
    document.getElementById(
        "modelAgreement"
    );

const totalTime =
    document.getElementById(
        "totalTime"
    );

const decisionMethod =
    document.getElementById(
        "decisionMethod"
    );


const modelResults =
    document.getElementById(
        "modelResults"
    );


const inputWordCount =
    document.getElementById(
        "inputWordCount"
    );


const analyzeAnotherButton =
    document.getElementById(
        "analyzeAnotherButton"
    );


/* ==========================================================
   WORD COUNT
========================================================== */

function countWords(text) {

    const cleanedText =
        text
            .trim()
            .replace(
                /\s+/g,
                " "
            );


    if (!cleanedText) {
        return 0;
    }


    return cleanedText
        .split(" ")
        .length;
}


function updateWordCount() {

    const articleWords =
        countWords(
            articleInput.value
        );


    const headlineWords =
        countWords(
            headlineInput.value
        );


    const totalWords =
        articleWords +
        headlineWords;


    wordCountElement.textContent =
        `${totalWords} ${totalWords === 1
            ? "word"
            : "words"
        }`;
}


/* ==========================================================
   NUMBER FORMATTERS
========================================================== */

function formatPercentage(
    value
) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(
            Number(value)
        )
    ) {

        return "N/A";
    }


    return `${(
        Number(value) * 100
    ).toFixed(2)}%`;
}


function formatTime(
    seconds
) {

    const value =
        Number(seconds);


    if (
        Number.isNaN(value)
    ) {

        return "N/A";
    }


    if (
        value < 1
    ) {

        return `${(
            value * 1000
        ).toFixed(0)} ms`;
    }


    return `${value.toFixed(2)} s`;
}


/* ==========================================================
   HTML ESCAPING
========================================================== */

function escapeHtml(
    value
) {

    const element =
        document.createElement(
            "div"
        );


    element.textContent =
        String(value);


    return element.innerHTML;
}


/* ==========================================================
   ERROR HANDLING
========================================================== */

function showError(
    message
) {

    errorMessage.textContent =
        message;

    errorMessage.classList.remove(
        "hidden"
    );
}


function clearError() {

    errorMessage.textContent =
        "";

    errorMessage.classList.add(
        "hidden"
    );
}


/* ==========================================================
   LOADING STATE
========================================================== */

function setLoading(
    loading
) {

    analyzeButton.disabled =
        loading;


    if (loading) {

        buttonText.textContent =
            "Analyzing...";

        buttonSpinner.classList.remove(
            "hidden"
        );

        loadingSection.classList.remove(
            "hidden"
        );

    } else {

        buttonText.textContent =
            "Analyze News";

        buttonSpinner.classList.add(
            "hidden"
        );

        loadingSection.classList.add(
            "hidden"
        );
    }
}


/* ==========================================================
   VALIDATE INPUT
========================================================== */

function validateInput() {

    const headline =
        headlineInput.value.trim();

    const article =
        articleInput.value.trim();


    const combined =
        `${headline} ${article}`
            .trim();


    if (!combined) {

        throw new Error(
            "Please enter a news article."
        );
    }


    const words =
        countWords(
            combined
        );


    if (
        words < MIN_WORDS
    ) {

        throw new Error(
            `Please enter at least ${MIN_WORDS} words. ` +
            `Current input contains ${words} words.`
        );
    }


    return {
        headline,
        article
    };
}


/* ==========================================================
   API ERROR MESSAGE
========================================================== */

function getApiErrorMessage(
    data,
    fallback
) {

    if (
        data &&
        typeof data.detail === "string"
    ) {

        return data.detail;
    }


    if (
        data &&
        Array.isArray(
            data.detail
        )
    ) {

        return data.detail
            .map(
                item =>
                    item.msg ||
                    "Invalid request"
            )
            .join(", ");
    }


    return fallback;
}


/* ==========================================================
   CALL PREDICTION API
========================================================== */

async function requestPrediction(
    headline,
    article
) {

    let response;


    try {

        response =
            await fetch(
                PREDICT_ENDPOINT,
                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            headline,
                            article
                        })
                }
            );

    } catch (error) {

        throw new Error(
            "Cannot connect to the prediction server. " +
            "Make sure the FastAPI backend is running on " +
            "http://127.0.0.1:8000."
        );
    }


    let data;


    try {

        data =
            await response.json();

    } catch (error) {

        throw new Error(
            "The server returned an invalid response."
        );
    }


    if (!response.ok) {

        throw new Error(
            getApiErrorMessage(
                data,
                `Prediction failed with status ${response.status}.`
            )
        );
    }


    return data;
}


/* ==========================================================
   CREATE MODEL RESULT CARD
========================================================== */

function createModelCard(
    model
) {

    const prediction =
        String(
            model.prediction
        );


    const predictionLower =
        prediction.toLowerCase();


    const confidenceWidth =
        Math.min(
            100,
            Math.max(
                0,
                Number(
                    model.confidence
                ) * 100
            )
        );


    const fakeWidth =
        Math.min(
            100,
            Math.max(
                0,
                Number(
                    model.fake_probability
                ) * 100
            )
        );


    const realWidth =
        Math.min(
            100,
            Math.max(
                0,
                Number(
                    model.real_probability
                ) * 100
            )
        );


    const card =
        document.createElement(
            "article"
        );


    card.className =
        "model-card";


    card.innerHTML = `

        <div class="model-card-header">

            <h3>
                ${escapeHtml(model.model)}
            </h3>

            <span
                class="
                    model-prediction-badge
                    ${predictionLower}
                "
            >
                ${escapeHtml(prediction)}
            </span>

        </div>


        <div class="probability">

            <div class="probability-label">

                <span>
                    Confidence
                </span>

                <strong>
                    ${formatPercentage(
        model.confidence
    )}
                </strong>

            </div>


            <div class="progress-track">

                <div
                    class="progress-value confidence"
                    style="
                        width:
                        ${confidenceWidth}%;
                    "
                ></div>

            </div>

        </div>


        <div class="probability">

            <div class="probability-label">

                <span>
                    Real probability
                </span>

                <strong>
                    ${formatPercentage(
        model.real_probability
    )}
                </strong>

            </div>


            <div class="progress-track">

                <div
                    class="progress-value real"
                    style="
                        width:
                        ${realWidth}%;
                    "
                ></div>

            </div>

        </div>


        <div class="probability">

            <div class="probability-label">

                <span>
                    Fake probability
                </span>

                <strong>
                    ${formatPercentage(
        model.fake_probability
    )}
                </strong>

            </div>


            <div class="progress-track">

                <div
                    class="progress-value fake"
                    style="
                        width:
                        ${fakeWidth}%;
                    "
                ></div>

            </div>

        </div>


        <p class="model-time">

            Prediction time:
            ${formatTime(
        model.time_seconds
    )}

        </p>
    `;


    return card;
}


/* ==========================================================
   DISPLAY MODEL RESULTS
========================================================== */

function renderModelResults(
    models
) {

    modelResults.innerHTML =
        "";


    if (
        !Array.isArray(models)
    ) {

        return;
    }


    models.forEach(
        model => {

            const card =
                createModelCard(
                    model
                );


            modelResults.appendChild(
                card
            );
        }
    );
}


/* ==========================================================
   DISPLAY FINAL RESULT
========================================================== */

function renderFinalResult(
    data
) {

    const final =
        data.final;


    const prediction =
        String(
            final.prediction
        );


    const predictionLower =
        prediction.toLowerCase();


    /* ------------------------------------------------------
       Reset result card classes
    ------------------------------------------------------ */

    finalResultCard.classList.remove(
        "real-result",
        "fake-result"
    );


    if (
        predictionLower === "real"
    ) {

        finalResultCard.classList.add(
            "real-result"
        );

    } else {

        finalResultCard.classList.add(
            "fake-result"
        );
    }


    /* ------------------------------------------------------
       Main result
    ------------------------------------------------------ */

    finalPrediction.textContent =
        prediction.toUpperCase();


    finalBadge.textContent =
        prediction.toUpperCase();


    finalConfidence.textContent =
        formatPercentage(
            final.confidence
        );


    selectedModel.textContent =
        final.selected_model;


    modelAgreement.textContent =
        final.agreement
            ? "Yes"
            : "No";


    totalTime.textContent =
        formatTime(
            data.total_time_seconds
        );


    decisionMethod.textContent =
        final.decision_method;


    inputWordCount.textContent =
        data.input?.word_count ?? 0;


    /* ------------------------------------------------------
       Model cards
    ------------------------------------------------------ */

    renderModelResults(
        data.models
    );


    /* ------------------------------------------------------
       Show result section
    ------------------------------------------------------ */

    resultSection.classList.remove(
        "hidden"
    );


    resultSection.scrollIntoView({
        behavior:
            "smooth",

        block:
            "start"
    });
}


/* ==========================================================
   ANALYZE NEWS
========================================================== */

async function analyzeNews() {

    clearError();


    resultSection.classList.add(
        "hidden"
    );


    let input;


    try {

        input =
            validateInput();

    } catch (error) {

        showError(
            error.message
        );

        return;
    }


    setLoading(
        true
    );


    try {

        const result =
            await requestPrediction(
                input.headline,
                input.article
            );


        renderFinalResult(
            result
        );

    } catch (error) {

        showError(
            error.message ||
            "Unable to analyze the news article."
        );

    } finally {

        setLoading(
            false
        );
    }
}


/* ==========================================================
   RESET
========================================================== */

function resetApplication() {

    headlineInput.value =
        "";

    articleInput.value =
        "";


    resultSection.classList.add(
        "hidden"
    );


    clearError();


    updateWordCount();


    window.scrollTo({
        top:
            0,

        behavior:
            "smooth"
    });


    headlineInput.focus();
}


/* ==========================================================
   BACKEND HEALTH CHECK
========================================================== */

async function checkBackendHealth() {

    try {

        const response =
            await fetch(
                HEALTH_ENDPOINT
            );


        if (!response.ok) {

            console.warn(
                "Backend health check failed."
            );

            return;
        }


        const health =
            await response.json();


        console.log(
            "Backend status:",
            health
        );

    } catch (error) {

        console.warn(
            "Backend is currently unavailable."
        );
    }
}


/* ==========================================================
   EVENTS
========================================================== */

articleInput.addEventListener(
    "input",
    updateWordCount
);


headlineInput.addEventListener(
    "input",
    updateWordCount
);


analyzeButton.addEventListener(
    "click",
    analyzeNews
);


analyzeAnotherButton.addEventListener(
    "click",
    resetApplication
);


/* ==========================================================
   CTRL + ENTER SHORTCUT
========================================================== */

articleInput.addEventListener(
    "keydown",
    event => {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            event.preventDefault();

            analyzeNews();
        }
    }
);


/* ==========================================================
   ULTRA-SMOOTH MATRIX GRID GLOW & WAVE SYSTEM
========================================================== */

function initMatrixGridGlow() {
    const canvas = document.getElementById("gridCanvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const CELL_SIZE = 36; // Matches 36px CSS grid
    let width = 0;
    let height = 0;
    let cols = 0;
    let rows = 0;

    let mouseX = -1000;
    let mouseY = -1000;
    let targetMouseX = -1000;
    let targetMouseY = -1000;
    let isHoveringInteractive = false;

    // Active cell intensity map (Key: "col_row" -> intensity float 0..1)
    const activeCells = new Map();

    // Active click shockwaves array: [{ x, y, radius, maxRadius, speed, width, intensity }]
    const clickRipples = [];

    function resizeCanvas() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        cols = Math.ceil(width / CELL_SIZE) + 1;
        rows = Math.ceil(height / CELL_SIZE) + 1;
    }

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    function activateGridAt(x, y) {
        const radiusPx = isHoveringInteractive ? 100 : 80;
        const baseIntensity = isHoveringInteractive ? 0.95 : 0.75;

        const startCol = Math.max(0, Math.floor((x - radiusPx) / CELL_SIZE));
        const endCol = Math.min(cols - 1, Math.ceil((x + radiusPx) / CELL_SIZE));
        const startRow = Math.max(0, Math.floor((y - radiusPx) / CELL_SIZE));
        const endRow = Math.min(rows - 1, Math.ceil((y + radiusPx) / CELL_SIZE));

        for (let c = startCol; c <= endCol; c++) {
            for (let r = startRow; r <= endRow; r++) {
                const cellCenterX = c * CELL_SIZE + CELL_SIZE / 2;
                const cellCenterY = r * CELL_SIZE + CELL_SIZE / 2;
                const distPx = Math.hypot(cellCenterX - x, cellCenterY - y);

                if (distPx <= radiusPx) {
                    const key = `${c}_${r}`;
                    const falloff = 0.5 + 0.5 * Math.cos((distPx / radiusPx) * Math.PI);
                    const current = activeCells.get(key) || 0;
                    const target = baseIntensity * falloff;
                    activeCells.set(key, Math.max(current, target));
                }
            }
        }
    }

    // Mouse tracking & grid cell activation only during movement
    window.addEventListener("pointermove", (e) => {
        activateGridAt(e.clientX, e.clientY);
    }, { passive: true });

    // Hover detection over interactive elements
    const interactiveSelector = "button, a, input, textarea, .brand-icon, .detail-item, .secondary-button, .model-card, .step-dot";

    document.addEventListener("mouseover", (e) => {
        if (e.target.closest(interactiveSelector)) {
            isHoveringInteractive = true;
        }
    });

    document.addEventListener("mouseout", (e) => {
        if (e.target.closest(interactiveSelector)) {
            isHoveringInteractive = false;
        }
    });

    // Create dynamic subtle continuous wave on click
    document.addEventListener("click", (e) => {
        clickRipples.push({
            x: e.clientX,
            y: e.clientY,
            radius: 0,
            maxRadius: isHoveringInteractive ? 140 : 100,
            speed: 5,
            width: 24,
            intensity: 0.5
        });
    });

    // Main 60fps animation loop
    function renderGrid() {
        ctx.clearRect(0, 0, width, height);

        // 2. Process active click shockwaves (subtle fluid energy wave)
        for (let i = clickRipples.length - 1; i >= 0; i--) {
            const ripple = clickRipples[i];
            ripple.radius += ripple.speed;
            const lifeProgress = ripple.radius / ripple.maxRadius;

            if (lifeProgress >= 1) {
                clickRipples.splice(i, 1);
                continue;
            }

            // Draw delicate expanding ring on canvas
            const waveOpacity = (1 - lifeProgress) * 0.18;
            ctx.strokeStyle = `rgba(52, 211, 153, ${waveOpacity})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(ripple.x, ripple.y, ripple.radius, 0, Math.PI * 2);
            ctx.stroke();

            // Light up grid cells passed over by the wavefront
            const startCol = Math.max(0, Math.floor((ripple.x - ripple.radius - ripple.width) / CELL_SIZE));
            const endCol = Math.min(cols - 1, Math.ceil((ripple.x + ripple.radius + ripple.width) / CELL_SIZE));
            const startRow = Math.max(0, Math.floor((ripple.y - ripple.radius - ripple.width) / CELL_SIZE));
            const endRow = Math.min(rows - 1, Math.ceil((ripple.y + ripple.radius + ripple.width) / CELL_SIZE));

            for (let c = startCol; c <= endCol; c++) {
                for (let r = startRow; r <= endRow; r++) {
                    const cx = c * CELL_SIZE + CELL_SIZE / 2;
                    const cy = r * CELL_SIZE + CELL_SIZE / 2;
                    const dist = Math.sqrt((cx - ripple.x) ** 2 + (cy - ripple.y) ** 2);
                    const distFromWave = Math.abs(dist - ripple.radius);

                    if (distFromWave <= ripple.width) {
                        const key = `${c}_${r}`;
                        const waveFactor = 0.5 + 0.5 * Math.cos((distFromWave / ripple.width) * Math.PI);
                        const waveGlow = waveFactor * (1 - lifeProgress) * 0.35;
                        const current = activeCells.get(key) || 0;
                        activeCells.set(key, Math.max(current, waveGlow));
                    }
                }
            }
        }

        // 3. Render all active glowing grid cells & intersection corner nodes
        activeCells.forEach((intensity, key) => {
            if (intensity <= 0.003) {
                activeCells.delete(key);
                return;
            }

            const parts = key.split("_");
            const col = parseInt(parts[0], 10);
            const row = parseInt(parts[1], 10);

            const x = col * CELL_SIZE;
            const y = row * CELL_SIZE;

            // Soft grid cell fill aura
            ctx.fillStyle = `rgba(16, 185, 129, ${intensity * 0.15})`;
            ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

            // Glowing grid border lines
            ctx.strokeStyle = `rgba(52, 211, 153, ${intensity * 0.55})`;
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 0.5, y + 0.5, CELL_SIZE, CELL_SIZE);

            // Corner glowing intersection node dots
            ctx.fillStyle = `rgba(52, 211, 153, ${intensity * 0.92})`;
            ctx.beginPath();
            ctx.arc(x, y, 1.8, 0, Math.PI * 2);
            ctx.fill();

            // Smooth decay per frame (0.945 for soft liquid dissolve)
            activeCells.set(key, intensity * 0.945);
        });

        requestAnimationFrame(renderGrid);
    }

    renderGrid();
}


/* ==========================================================
   APP FULL-SCREEN PRELOADER CONTROLLER
========================================================== */

function initAppPreloader(onComplete) {
    const preloader = document.getElementById("appPreloader");
    const fill = document.getElementById("preloaderBar");
    const percentText = document.getElementById("preloaderPercent");
    const statusText = document.getElementById("preloaderStatusText");
    const appContainer = document.querySelector(".app-container");

    document.body.classList.add("is-loading");

    if (!preloader || !fill || !percentText || !statusText) {
        document.body.classList.remove("is-loading");
        if (appContainer) appContainer.classList.add("page-loaded");
        if (typeof onComplete === "function") onComplete();
        return;
    }

    let isDismissed = false;

    function dismissPreloader() {
        if (isDismissed) return;
        isDismissed = true;

        document.body.classList.remove("is-loading");
        if (appContainer) appContainer.classList.add("page-loaded");

        preloader.classList.add("fade-out");
        preloader.style.pointerEvents = "none";

        setTimeout(() => {
            if (preloader && preloader.parentNode) {
                preloader.parentNode.removeChild(preloader);
            }
            if (typeof onComplete === "function") onComplete();
        }, 500);
    }

    // Fail-safe timeout: Force dismiss after 5.5s maximum
    const failSafeTimer = setTimeout(dismissPreloader, 5500);

    const stages = [
        { progress: 25, text: "Initializing Verification Core..." },
        { progress: 55, text: "Connecting to Inference Engine..." },
        { progress: 85, text: "Loading Random Forest & BERT Models..." },
        { progress: 100, text: "System Ready" }
    ];

    let currentStage = 0;
    let progress = 0;
    const TOTAL_DURATION_MS = 4000;
    const STEP_INTERVAL_MS = 38;
    const startTime = Date.now();

    const interval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        progress = Math.min(100, Math.floor((elapsed / TOTAL_DURATION_MS) * 100));

        if (currentStage < stages.length && progress >= stages[currentStage].progress) {
            statusText.textContent = stages[currentStage].text;
            currentStage++;
        }

        fill.style.width = `${progress}%`;
        percentText.textContent = `${progress}%`;

        if (progress >= 100) {
            clearInterval(interval);
            clearTimeout(failSafeTimer);
            setTimeout(dismissPreloader, 250);
        }
    }, STEP_INTERVAL_MS);
}


/* ==========================================================
   SMOOTH LIGHT / DARK THEME CONTROLLER
========================================================== */

function initThemeToggle() {
    const toggleBtn = document.getElementById("themeToggleBtn");
    const savedTheme = localStorage.getItem("theme") || "dark";

    document.documentElement.setAttribute("data-theme", savedTheme);

    if (!toggleBtn) return;

    toggleBtn.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme");
        const newTheme = currentTheme === "light" ? "dark" : "light";

        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
    });
}


/* ==========================================================
   STARTUP
========================================================== */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        initThemeToggle();

        updateWordCount();

        checkBackendHealth();

        initMatrixGridGlow();

        initAppPreloader(() => {
            headlineInput.focus();
        });
    }
);
