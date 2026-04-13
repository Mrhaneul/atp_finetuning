# CAIMLL ATP 2-01.3 Fine-Tuning Strategy — FINAL
## Gemma 4 31B Dense — Thinking-Enabled Doctrine Reasoning Model

**Date:** April 12, 2026  
**Version:** 2.0 (consolidated from initial strategy + revision session)  
**Project:** CAIMLL Fine-Tuning  
**Target Model:** Gemma 4 31B Dense (30.7B params, Apache 2.0)  
**Generator Model:** Gemma 4 31B-IT (same model, thinking mode enabled)  
**Source Document:** ATP 2-01.3, Intelligence Preparation of the Battlefield (March 2019)  
**Platform:** DGX Spark GB10 (~119GB VRAM)  
**Presentation Deadline:** This week

---

## 1. Problem Statement

Previous CAIMLL pipelines (FM 2-0, V1 through V2b) used Alpaca-style question-answer pairs to fine-tune Llama 3.2 3B. The best result (V2b, eval score 0.487) revealed a core limitation: the model learned doctrinal vocabulary and response style but not deep doctrinal reasoning. It sounds like a knowledgeable assistant but cannot reliably reproduce specific enumerations, distinguish echelon-specific roles, or apply doctrine to novel scenarios.

Moving to Gemma 4 31B with its native thinking mode presents an opportunity to solve this at the dataset level. Instead of just teaching the model what the doctrine says, we teach it how to reason through the doctrine — grounding every answer in paragraph-level citations, multi-step doctrinal logic, and the kind of applied reasoning that trained intelligence analysts perform naturally.

### Goals

1. **Recall** — accurately reproduce doctrinal definitions, processes, and enumerations from ATP 2-01.3
2. **Reason** — walk through multi-step doctrinal logic (e.g., "If the terrain is urban, how does IPB Step 2 change?")
3. **Apply** — generate doctrinally grounded analysis for scenarios not explicitly covered in the manual (e.g., constructing an HVT list for a peer threat in a littoral environment)
4. **Cite** — anchor every claim to specific ATP 2-01.3 paragraphs, chapters, and steps (citations at END of answer, not beginning)

---

## 2. Model Selection

### Fine-Tune Target: Gemma 4 31B Dense

Released April 2, 2026. Apache 2.0 (American origin — Google DeepMind).

Key specs:
- 30.7B parameters, dense architecture (not MoE — better for SFT)
- Native thinking mode via `<|think|>` token → `<|channel>thought\n[reasoning]<channel|>[answer]`
- Native system prompt support (standard system/assistant/user roles)
- 256K context window
- AIME 2026: 89.2%, GPQA Diamond: 84.3%, MMLU Pro: 85.2%
- Fits under ~22GB VRAM with Unsloth QLoRA 4-bit
- Apache 2.0 — no licensing restrictions for military/government deployment
- #3 open model worldwide on Arena AI text leaderboard

### Generator Model: Gemma 4 31B-IT (same model, thinking enabled)

Previous plan used gpt-oss:120b (OpenAI). Switching to Gemma 4 31B-IT for generation based on benchmark analysis:

| Benchmark | Gemma 4 31B | gpt-oss:120b | Delta |
|-----------|-------------|--------------|-------|
| GPQA Diamond | 84.3% | 76.2% | Gemma 4 +8.1 |
| AIME 2026 | 89.2% | ~83% | Gemma 4 +6 |
| MMLU Pro | 85.2% | ~80% | Gemma 4 +5 |
| LiveCodeBench v6 | 80.0% | ~72% | Gemma 4 +8 |

Additional advantages of using Gemma 4 31B as generator:
- Native `<|think|>` mode produces structured reasoning chains without prompt engineering
- Eliminates VRAM contention (no need to juggle two models — generate first, then train)
- Same tokenizer and chat template for generation and training = no format mismatch
- American origin, Apache 2.0 — same provenance as the target model

Self-distillation concern is mitigated because: the doctrine text IS the knowledge source. The model reads specific ATP paragraphs and generates structured QA from them. This is grounded reading comprehension with synthesis, not pure knowledge distillation.

### Why NOT these alternatives

| Model | Origin | Why Not |
|-------|--------|---------|
| gpt-oss:120b | OpenAI (US) | Weaker reasoning (GPQA 76.2 vs 84.3); VRAM contention with training |
| GLM-5 | Zhipu AI (China) | Excluded — Chinese origin |
| Qwen 3.5 | Alibaba (China) | Excluded — Chinese origin |
| DeepSeek V3.2 | DeepSeek (China) | Excluded — Chinese origin |
| Kimi K2.5 | Moonshot AI (China) | Excluded — Chinese origin |
| Llama 4 Scout | Meta (US) | Needs ~55GB INT4 min; custom license (not Apache 2.0); GPQA 74.3% (weaker) |
| Llama 4 Maverick | Meta (US) | Needs 8x H100 — won't fit on DGX Spark |

Fallback: If Gemma 4 31B generation quality disappoints on early samples, revert to gpt-oss:120b (already deployed on the DGX Spark via Ollama).

---

## 3. Pipeline Architecture — Five Stages

SME review pipeline is designed but deferred to Phase 2 (post-presentation).

```
┌──────────────────────────────────────────────────────────────────┐
│                     ATP 2-01.3 PDF                               │
└──────────────┬───────────────────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │  STAGE 1        │
       │  Parse & Chunk  │  pdftotext → paragraph-ID-aware chunking
       │                 │  → chunks.jsonl
       └───────┬─────────┘
               │
       ┌───────▼────────┐
       │  STAGE 2        │
       │  Metadata       │  Gemma 4 31B classifies each chunk:
       │  Enrichment     │  IPB step, chapter, content type, echelon,
       │  (grounded)     │  domain, threat type — structural only
       └───────┬─────────┘
               │
       ┌───────▼────────┐
       │  STAGE 3        │
       │  Synthetic QA   │  Multi-type question generation with
       │  + Thinking     │  <think> reasoning traces
       │  Traces         │  Target: 5K initial → scale to 10K
       └───────┬─────────┘
               │
       ┌───────▼────────┐
       │  STAGE 4        │
       │  Format &       │  Gemma 4 chat template, quality filter,
       │  Train          │  chapter-weighted sampling, QLoRA SFT
       └───────┬─────────┘
               │
       ┌───────▼────────┐
       │  STAGE 5        │
       │  Evaluate &     │  ATP-specific eval suite (24 questions),
       │  Iterate        │  DPO on weak questions
       └──────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  PHASE 2 (post-presentation, when SMEs available)   │
  │  ├── SME review interface deployment                │
  │  ├── SME validation of synthetic data               │
  │  ├── SME-contributed gold examples                  │
  │  ├── DPO on weak questions using SME pairs          │
  │  └── Re-train with SME-validated dataset            │
  └─────────────────────────────────────────────────────┘
```

---

## 4. Stage 1 — Parse & Chunk with Paragraph-ID Awareness

ATP 2-01.3 uses standard Army paragraph numbering (e.g., `1-1.`, `3-24.`, `6-15.`). Paragraph IDs are preserved as first-class metadata.

### Chunk schema (chunks.jsonl)

```json
{
  "chunk_id": "atp201_3-ch3-p24",
  "para_id": "3-24",
  "text": "The full paragraph text...",
  "chapter_num": 3,
  "chapter_title": "Step 1—Define the Operational Environment",
  "section": "Identify the Significant Characteristics of the OE",
  "page_range": [42, 43],
  "word_count": 187,
  "has_figure_reference": false,
  "cross_references": ["ATP 2-01.3, para 1-4", "FM 2-0"]
}
```

### Extraction

```bash
pdftotext ATP_2-01_3_Intelligence_Preparation_of_the_Battlefield.pdf atp_raw.txt
```

No `-layout` flag (per DGX-proven approach). Chunker regex:

```python
PARA_PATTERN = re.compile(r'^(\d+-\d+)\.\s')
```

---

## 5. Stage 2 — Metadata Enrichment (Grounded Only)

Metadata is strictly structural classification of content that already exists in the manual. It does NOT add new facts, examples, or beyond-manual content. Think of it as a librarian adding catalog tags — the content doesn't change.

### What metadata IS (safe, grounded)

Every field can be verified by reading the paragraph:

| Field | Values | Verification |
|-------|--------|-------------|
| `ipb_step` | 0 (fundamentals), 1, 2, 3, 4, null | Chapter 3 = Step 1, Chapter 4 = Step 2, etc. |
| `content_type` | definition, process, product, example, checklist, considerations, cross_reference | Readable from paragraph structure |
| `echelon` | tactical, operational, strategic, multi | Stated or implied in text |
| `domain` | land, air, maritime, space, cyberspace, information, multi_domain | Stated in text |
| `threat_type` | regular, irregular, hybrid, peer, null | Stated in text |
| `ipb_product` | MCOO, sittemp, event_template, HVT_list, terrain_effects_matrix, weather_effects, civil_overlay, null | Named in text |
| `environment` | general, urban, littoral, subterranean, null | Chapter 7 context |
| `doctrinal_weight` | 1.0-3.0 | Adjustable importance score |

### What metadata IS NOT

- Invented examples ("in Iraq, this would look like...")
- Doctrinal opinions ("this is the most important step")
- Cross-manual references the manual doesn't make
- Tactical scenarios not in the text
- SME commentary or real-world context

"Beyond the manual" reasoning capability comes from QUESTION DESIGN in Stage 3, not from metadata injection.

### Classification prompt (Gemma 4 31B-IT)

```
<|system|>
<|think|>
You are a military intelligence doctrine classifier. Given a paragraph from 
ATP 2-01.3, classify it with structural metadata. Respond ONLY in valid JSON.
<|end|>
<|user|>
PARAGRAPH:
{chunk_text}

CHAPTER: {chapter_num} — {chapter_title}
PARAGRAPH ID: {para_id}

Classify this paragraph:
{
  "ipb_step": <0|1|2|3|4|null>,
  "content_type": <"definition"|"process"|"product"|"example"|"checklist"|"considerations"|"cross_reference">,
  "echelon": <"tactical"|"operational"|"strategic"|"multi">,
  "domain": <"land"|"air"|"maritime"|"space"|"cyberspace"|"information"|"multi_domain">,
  "threat_type": <"regular"|"irregular"|"hybrid"|"peer"|null>,
  "ipb_product": <product name or null>,
  "environment": <"general"|"urban"|"littoral"|"subterranean"|null>,
  "doctrinal_weight": <1.0-3.0>
}
<|end|>
```

### Quality gate

- Chapter 3 chunks must have `ipb_step: 1`
- Chapter 4 chunks must have `ipb_step: 2`
- Chapter 5 chunks must have `ipb_step: 3`
- Chapter 6 chunks must have `ipb_step: 4`
- Reject any chunk where `ipb_step` conflicts with chapter number

---

## 6. Stage 3 — Synthetic QA Generation with Thinking Traces

### Training example schema

```json
{
  "qa_id": "atp201_qa_0042",
  "source_chunks": ["atp201_3-ch3-p24", "atp201_3-ch3-p25"],
  "question_type": "applied_reasoning",
  "question": "A BCT S-2 is conducting IPB for an operation in a densely urbanized area. How should Step 2 be modified compared to open terrain?",
  "thinking_trace": "Let me work through this using ATP 2-01.3 doctrine.\n\n1. Step 2 of IPB focuses on describing how the OE affects both friendly and threat operations (Chapter 4).\n\n2. The standard approach involves evaluating OAKOC — observation and fields of fire, avenues of approach, key terrain, obstacles, and cover and concealment (para 4-12 through 4-30).\n\n3. Chapter 7 addresses IPB in urban environments specifically. Urban terrain fundamentally changes OAKOC analysis:\n   - Observation is restricted to line-of-sight between buildings (para 7-27)\n   - Fields of fire become three-dimensional: rooftops, upper floors, subterranean (para 7-28)\n   - Avenues of approach include interior building routes, sewer systems, and rooftops\n   - Key terrain shifts to critical infrastructure: bridges, communications nodes, government buildings\n\n4. The MCOO must become three-dimensional rather than traditional 2D.\n\n5. Civil considerations are elevated — population density, displaced persons, critical infrastructure.\n\nSynthesizing: Urban IPB Step 2 requires 3D OAKOC, elevated civil considerations, and modified products.",
  "answer": "When conducting IPB Step 2 in a densely urbanized area, the BCT S-2 must make several key modifications compared to open terrain.\n\nThe OAKOC analysis must become three-dimensional. Urban terrain transforms each factor: observation is restricted to line-of-sight between structures, fields of fire extend vertically across rooftops and upper floors as well as subterranean levels, and avenues of approach expand to include interior building routes, sewer systems, and rooftop pathways.\n\nThe Modified Combined Obstacle Overlay must account for vertical and subterranean dimensions rather than the traditional 2D overlay used in open terrain. Key terrain identification shifts from dominant high ground to critical infrastructure such as bridges, communications nodes, and government buildings.\n\nCivil considerations are significantly elevated. Population density, displaced persons flow, critical civilian infrastructure, and cultural or religious sites all create operational constraints that rarely apply at the same scale in open terrain.\n\nFinally, the terrain effects matrix requires additional analysis categories for structural integrity, vertical access points, and subterranean connectivity.\n\n[Reference: ATP 2-01.3, Ch. 4 (IPB Step 2), Ch. 7 (Urban environments), para 4-12 through 4-30, para 7-25 through 7-35]",
  "metadata": {
    "ipb_step": 2,
    "content_type": "applied_reasoning",
    "echelon": "tactical",
    "environment": "urban",
    "difficulty": "hard"
  }
}
```

### Citation format — answer first, references at end

The thinking trace cites heavily throughout (that's where grounding happens). The final answer is written in natural analyst voice with a reference block at the end.

Why NOT "According to ATP 2-01.3..." at the start:
- Risk of fabricated citations — model generates left-to-right, committing to a paragraph number before generating the substance
- Unnatural voice — real analysts don't talk like footnotes
- If a citation is wrong, it's baked into the substance of the answer

Format: THINK with citations → SPEAK naturally → CITE at the bottom.

### Question type taxonomy

| Type | Description | Think Depth | Target % |
|------|-------------|-------------|----------|
| `factual` | Direct recall of definitions, terms, lists | Short (2-3 steps) | 15% |
| `procedural` | Step-by-step process execution | Medium (4-6 steps) | 15% |
| `comparative` | Compare/contrast concepts, products, steps | Medium (4-6 steps) | 10% |
| `echelon_specific` | Role differences across BCT/Division/Corps | Medium, must distinguish | 10% |
| `applied_reasoning` | Apply doctrine to novel scenario | Deep (6-10 steps) | 20% |
| `product_generation` | Create/describe an IPB product | Deep, structured output | 10% |
| `cross_step` | How output of one IPB step feeds another | Deep, shows flow | 10% |
| `multi_domain` | IPB across land/air/cyber/space/info domains | Deep, Ch 8 focus | 5% |
| `contrastive` | "Why X and not Y" — forces precise distinction | Medium, high precision | 5% |

### How "beyond the manual" reasoning works

The model learns to think beyond the manual through QUESTION DESIGN, not metadata. The pattern:

```
DOCTRINE SAYS: "A high-value target is a target the enemy commander 
requires for the successful completion of the mission." (para 6-XX)

THE MANUAL DOES NOT GIVE: A specific HVT example for a peer threat 
in a littoral environment.

THE QUESTION FORCES APPLICATION:
"Based on the doctrinal definition of high-value targets, what would 
you include on the initial HVT list for a peer threat with integrated 
A2/AD in a contested littoral environment, and why?"

THE THINKING TRACE BRIDGES:
"The manual defines HVTs as targets the enemy commander requires for 
mission success (para 6-XX). In a contested littoral environment 
against a peer threat, I need to identify what that commander requires:
- Anti-ship cruise missile batteries → sea denial requires these
- IADS nodes → protecting the littoral zone requires air defense
- Naval mine-laying assets → area denial requires these
- C2 nodes → coordinating joint fires requires these
Each meets the doctrinal definition: the enemy commander REQUIRES 
them for mission success."
```

The model learns: "I know what an HVT IS from doctrine → I can APPLY that definition to any scenario." It never invents doctrine. It applies doctrine.

### Generation prompt (Gemma 4 31B-IT with thinking enabled)

```
<|system|>
<|think|>
You are a senior military intelligence analyst and doctrine instructor 
generating training data for a reasoning AI. Generate a question, a 
detailed thinking trace, and a final answer based on the provided 
ATP 2-01.3 content. Your thinking trace must reference specific paragraph 
numbers and show multi-step doctrinal logic. Your answer must be written 
in natural analyst voice with a [Reference: ...] block at the end.
<|end|>
<|user|>
SOURCE CONTENT:
{chunk_text}

METADATA:
- Chapter: {chapter_num} — {chapter_title}
- IPB Step: {ipb_step}
- Paragraph(s): {para_ids}
- Content Type: {content_type}
- Environment: {environment}

QUESTION TYPE TO GENERATE: {question_type}

Generate in this exact JSON format:
{
  "question": "...",
  "thinking_trace": "...",
  "answer": "...",
  "difficulty": "easy|medium|hard",
  "citation_paragraphs": ["3-24", "3-25"]
}
<|end|>
```

### Question diversity strategy

For each doctrine concept/paragraph, generate across multiple question types:

```
Paragraph 3-24 (defines operational environment characteristics):
├── Factual: "What are the significant characteristics of the OE?"
├── Procedural: "Walk through the steps to identify OE characteristics"
├── Applied: "How would you identify OE characteristics for a 
│             peacekeeping operation in sub-Saharan Africa?"
├── Comparative: "How does defining the OE differ between offense 
│                 and stability operations?"
├── Contrastive: "Why is defining the OE the first IPB step and 
│                 not threat evaluation?"
├── Cross-step: "How do the OE characteristics identified in Step 1 
│                shape your terrain analysis in Step 2?"
├── Echelon: "How does the scope of OE definition differ between 
│             BCT S-2 and Division G-2?"
└── Multi-domain: "What cyberspace and information environment 
                   factors must be included when defining the OE?"
```

8 questions × ~250 key chunks = ~2,000 core examples. Then 2-3 scenario variants of applied/comparative/contrastive questions = 5,000-10,000 total.

### Volume target

| Phase | Count | Purpose |
|-------|-------|---------|
| Phase 1 (this week) | ~5,000 QA triples | Initial train run + eval |
| Phase 2 (if Phase 1 loss is healthy) | Scale to ~10,000 | More diversity, stronger model |
| After SME review | Replace/upgrade weak examples | Gold-standard refinement |

---

## 7. Stage 4 — Training Format & Execution

### Gemma 4 chat template with thinking

```
<bos><|system|>
<|think|>
You are a military intelligence analyst with deep expertise in ATP 2-01.3, 
Intelligence Preparation of the Battlefield. This question concerns 
{ipb_step_name} (IPB Step {ipb_step}), Chapter {chapter_num}: {chapter_title}.
Echelon context: {echelon}. Environment: {environment}.

Always reason step by step through the relevant doctrine before answering.
Cite specific ATP 2-01.3 paragraphs in your reasoning. Place reference 
citations at the end of your answer.
<|end|>
<|user|>
{question}
<|end|>
<|assistant|>
<|channel>thought
{thinking_trace}
<channel|>
{answer}
<|end|>
```

### Training hyperparameters

| Parameter | 5K dataset | 10K dataset | Rationale |
|-----------|-----------|------------|-----------|
| Base model | google/gemma-4-31B-it (or Unsloth 4-bit) | same | Dense, best for SFT |
| Quantization | QLoRA 4-bit (bnb) | same | ~22GB VRAM |
| LoRA r | 16 | 16 (or 32 if capacity allows) | Proven conservative at V2b |
| LoRA alpha | 32 | 32 (or 64) | 2x r |
| LoRA targets | q,k,v,o,gate,up,down | same | All linear layers |
| Epochs | 2-3 | 1-2 | Fewer epochs at higher data volume |
| Learning rate | 2e-4 | 1e-4 to 2e-4 | Slightly lower at 10K |
| Batch size | 2 | 2 | VRAM constrained |
| Grad accum | 8 | 8 | Effective batch = 16 |
| Max seq length | 4096 | 4096 | Thinking traces are longer than Alpaca |
| Optimizer | adamw_8bit | same | Memory efficient |
| LR scheduler | cosine | same | Proven |
| Target train loss | ~1.0-1.2 | ~1.0-1.3 | Sweet spot from V2b |

### Chapter-weighted sampling

```python
CHAPTER_WEIGHTS = {
    1: 2.5,   # IPB fundamentals, definitions
    2: 2.0,   # IPB support to decision making / MDMP
    3: 2.0,   # Step 1 — Define the OE
    4: 2.0,   # Step 2 — Describe Environmental Effects
    5: 2.5,   # Step 3 — Evaluate the Threat
    6: 2.5,   # Step 4 — Determine Threat COAs (HVTs, sit templates)
    7: 3.0,   # Offense/defense/stability + urban/littoral/subterranean
    8: 2.0,   # Multi-domain considerations
    "A": 1.5,  # Appendix A — IPB checklist
    "B": 1.5,  # Appendix B — analyst tools
    "C": 2.0,  # Appendix C — threat characteristics
    "D": 1.5,  # Appendix D — cyberspace integration
}
```

### Execution sequence (no VRAM contention)

```bash
# === GENERATION PHASE (Stages 1-3) ===
# Load Gemma 4 31B-IT via Ollama for generation
ollama pull gemma4:31b

# Run chunker, enricher, generator sequentially
python run_pipeline.py --stages 1,2,3

# === TRAINING PHASE (Stages 4-5) ===
# Unload generator
ollama stop gemma4:31b

# Verify VRAM is free
nvidia-smi

# Run formatter + trainer
python run_pipeline.py --stages 4,5

# === EVAL PHASE ===
# Evaluate (can reload Ollama with fine-tuned adapter via GGUF)
python run_pipeline.py --stages eval
```

---

## 8. Stage 5 — Evaluation

### ATP 2-01.3 eval suite (24 questions)

| ID | Type | IPB Step | Question Focus |
|----|------|----------|----------------|
| F-01 | factual | 0 | IPB definition and purpose |
| F-02 | factual | 0 | IPB products (list all five categories) |
| F-03 | factual | 1 | Area of interest vs area of operations |
| F-04 | factual | 3 | Definition of HVT vs HPT |
| P-01 | procedural | 1 | Steps to define the operational environment |
| P-02 | procedural | 2 | How to construct an MCOO |
| P-03 | procedural | 4 | Steps to develop threat COAs |
| P-04 | procedural | 0 | How IPB integrates with MDMP |
| C-01 | comparative | 2-3 | Terrain analysis vs threat evaluation |
| C-02 | comparative | 3 | Regular vs irregular vs hybrid threat |
| E-01 | echelon | 0 | BCT S-2 vs Division G-2 IPB responsibilities |
| E-02 | echelon | 0 | How IPB products differ at tactical vs operational |
| AR-01 | applied | 2 | Modify terrain analysis for urban environment |
| AR-02 | applied | 3 | Evaluate peer threat with A2/AD capabilities |
| AR-03 | applied | 4 | Generate situation template for hybrid threat |
| AR-04 | applied | 2 | Weather effects analysis for airborne operation |
| PG-01 | product_gen | 4 | Create an HVT list for a given scenario |
| PG-02 | product_gen | 4 | Describe components of an event template |
| CS-01 | cross_step | 1→2 | How Step 1 outputs feed Step 2 |
| CS-02 | cross_step | 3→4 | How threat evaluation informs COA development |
| MD-01 | multi_domain | 2 | Cyberspace effects on terrain analysis |
| MD-02 | multi_domain | 3 | Space domain threat capabilities in IPB |
| CT-01 | contrastive | 4 | Why situation templates and not just threat COA statements |
| CT-02 | contrastive | 2 | MCOO vs terrain effects matrix — when to use which |

### Scoring methodology

**Keyword coverage** (automated):
- Extract doctrinal keywords per question from source paragraphs
- Score = (keywords present in answer) / (total expected keywords)

**Citation accuracy** (automated):
- Are cited paragraphs real ATP 2-01.3 paragraph numbers?
- Do cited paragraphs actually contain the claimed content? (match against chunks.jsonl)

**Reasoning quality** (automated + future SME):
- Does thinking trace reference correct paragraph numbers?
- Does reasoning follow logical chain?
- Does conclusion follow from reasoning?

### DPO on weak questions (post-SFT)

After initial eval, identify questions scoring below 0.5:

```json
{
  "prompt": "{question}",
  "chosen": "{correct answer with proper citations}",
  "rejected": "{the model's actual incorrect output from eval}"
}
```

---

## 9. Dataset Composition Summary

| Category | Count (5K) | Count (10K) | % |
|----------|-----------|------------|-----|
| Factual (short think traces) | ~750 | ~1,500 | 15% |
| Procedural (medium think traces) | ~750 | ~1,500 | 15% |
| Comparative (medium think traces) | ~500 | ~1,000 | 10% |
| Echelon-specific (medium think traces) | ~500 | ~1,000 | 10% |
| Applied reasoning (deep think traces) | ~1,000 | ~2,000 | 20% |
| Product generation (deep think traces) | ~500 | ~1,000 | 10% |
| Cross-step reasoning (deep think traces) | ~500 | ~1,000 | 10% |
| Multi-domain (deep think traces) | ~250 | ~500 | 5% |
| Contrastive (medium think traces) | ~250 | ~500 | 5% |
| **Total** | **~5,000** | **~10,000** | **100%** |

After quality filtering (expect ~15% rejection rate):
- **5K run**: ~4,250 train / ~470 val (10% split)
- **10K run**: ~8,500 train / ~940 val
- **24 held-out eval questions** (never seen during training)

---

## 10. SME Pipeline — Phase 2 Design (Deferred)

Designed and documented for presentation. Executes when SME veterans are available.

### SME contributions (two types)

1. **Validation** — Review synthetic QA triples for doctrinal accuracy, reasoning quality, citation correctness. Rate 1-5. Approve / Approve with edits / Reject.
2. **Gold examples** — SMEs contribute their own questions and answers in their natural style. These become highest-weight training examples.

### SME review interface

Web-based tool presenting one QA triple at a time with:
- Question display
- Editable thinking trace
- Editable answer
- Read-only source paragraphs for verification
- Rating scales (doctrinal accuracy, reasoning quality, citation accuracy)
- Flags (hallucinated, wrong para ref, incomplete, needs rewrite)
- Approve / Edit / Reject buttons

### SME workflow priority

1. Applied reasoning + contrastive questions first (highest hallucination risk)
2. Spot-check factual + procedural (20% sample after automated para-matching)
3. Target: 500+ SME-reviewed examples before production deployment

### Integration with synthetic pipeline

SME contributions replace or upgrade synthetic examples — they don't need to be present for the initial pipeline to run. The pipeline is designed for SME input to be additive.

---

## 11. File Structure

```
~/caimll_finetuning/atp_pipeline_v2/
├── chunker.py          # Stage 1: Parse PDF → chunks.jsonl
├── enricher.py         # Stage 2: Metadata classification (grounded)
├── generator.py        # Stage 3: Thinking-trace QA generation
├── formatter.py        # Stage 4: Gemma 4 chat template formatting
├── trainer.py          # Stage 4: QLoRA SFT training
├── eval.py             # Stage 5: Automated evaluation
├── dpo.py              # Stage 5: DPO on weak questions
├── run_pipeline.py     # Orchestrator with stage skipping
├── monitor.py          # Live progress monitoring
├── burn_gguf.py        # GGUF export for Ollama deployment
├── data/
│   ├── chunks.jsonl    # Raw chunks with paragraph IDs
│   ├── enriched.jsonl  # Chunks + grounded metadata
│   ├── seeds.jsonl     # Generated QA triples with thinking traces
│   ├── train.jsonl     # Formatted training data
│   └── val.jsonl       # Validation split
├── eval/
│   ├── questions.json  # 24 eval questions
│   └── results/        # Per-run eval results
└── outputs/
    └── atp-gemma4-lora/ # Trained adapter
```

---

## 12. Execution Timeline (This Week)

| Day | Task | Output |
|-----|------|--------|
| Day 1 | Stage 1: Parse & chunk ATP 2-01.3 | chunks.jsonl |
| Day 1 | Stage 2: Metadata enrichment | enriched.jsonl |
| Day 1-2 | Stage 3: Generate 5K QA triples | seeds.jsonl |
| Day 2 | Quality filter + format | train.jsonl, val.jsonl |
| Day 2-3 | Stage 4: QLoRA SFT on Gemma 4 31B | adapter checkpoint |
| Day 3 | Stage 5: Evaluate against 24-question suite | eval results |
| Day 3-4 | If loss healthy: scale to 10K, retrain | improved adapter |
| Day 4 | DPO on weak questions if needed | refined adapter |
| Day 4 | GGUF export for demo | Ollama-ready model |
| Day 5 | Presentation prep | demo + strategy doc |

---

## 13. Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| Overall keyword coverage (24 eval Qs) | >0.65 | >0.75 |
| Factual question accuracy | >0.85 | >0.95 |
| Applied reasoning quality (future SME-scored) | >3.5/5.0 | >4.0/5.0 |
| Citation accuracy (paragraph refs) | >0.80 | >0.90 |
| Echelon distinction (E-01, E-02) | >0.60 | >0.75 |
| No doctrinal hallucination in eval | <2/24 Qs | 0/24 Qs |
| Train loss | 1.0-1.2 | ~1.0 |

---

## 14. Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Gemma 4 31B QLoRA doesn't fit on DGX Spark | Unsloth confirms ~22GB; fallback to Gemma 3 27B |
| Self-distillation ceiling (same model generates and trains) | Doctrine text is the knowledge source, not the model; optional: use proprietary API for hardest 500 examples |
| Gemma 4 31B generates poor thinking traces | Test on 50 examples first; revert to gpt-oss:120b if quality disappoints |
| Thinking traces too long → loss explodes | Cap at 500 tokens during generation; filter outliers |
| Model memorizes thinking patterns not reasoning | Diverse scenarios per chunk (8 question types); multiple scenario variants |
| Echelon blurring (FM 2-0 lesson) | Metadata-enriched system prompts + contrastive questions + echelon-specific eval |
| 10K examples overfit | Start at 5K, check loss; reduce epochs to 1-2 at 10K |
| Fabricated citations | Citations at end (not beginning); automated para-ID validation against chunks.jsonl |

---

## 15. Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Gemma 4 31B over gpt-oss:120b for generation | +8 GPQA Diamond, native thinking mode, eliminates VRAM contention |
| Gemma 4 31B Dense over 26B MoE for fine-tuning | MoE expert routing can be disrupted by domain SFT; dense is safer |
| Citations at END not beginning | Prevents fabricated citations; more natural analyst voice |
| Metadata = structural only | Beyond-manual content via question design, not metadata injection |
| SME pipeline deferred to Phase 2 | This week's deadline; synthetic pipeline runs independently |
| 5K→10K dataset scaling | 10K is sweet spot for domain knowledge on 31B model; start 5K to validate |
| No Chinese-origin models | Project constraint; all models are American origin + Apache 2.0 |

---

*This document is the technical blueprint for the ATP 2-01.3 phase of CAIMLL.
All decisions are grounded in FM 2-0 V1/V2/V2b lessons learned, current 
benchmark data (April 2026), and chain-of-thought fine-tuning best practices.*
