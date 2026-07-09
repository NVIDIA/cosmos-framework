# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canonical V4.2 upsampler prompt templates — inference-team entry point.

Standalone module. No file I/O, no non-stdlib imports. The canonical templates
for each task are inlined as Python string literals. To extend with a new
version, add an entry to ``CANONICAL_TEMPLATES`` keyed by ``(version, task)``.

Usage::

    from cosmos_framework.model.generator.upsampler.prompts import (
        build_user_text, build_messages, clean_response, is_upsampled_prompt,
    )

    # Text-to-video (video parameters required)
    user_text = build_user_text(
        task="t2v", description="A cat playing with yarn",
        fps=24, duration_secs=6, aspect_ratio="16,9",
        resolution_w=1280, resolution_h=720,
    )

    # Text-to-image: the v4.2 default is the EXPRESSIVE body — v4.2 structure
    # plus a "fill plausibly, leave empties only when truly non-applicable"
    # rule.  Use without a version override unless you want a different t2i
    # variant (see below).
    user_text = build_user_text(
        task="t2i", description="A photo of a corgi",
        aspect_ratio="1,1", resolution_w=960, resolution_h=960,
    )

    # Text-to-image, original v4.2 baseline body (kept for A/B comparisons
    # against the expressive default — e.g. UGB baseline MR-366).
    user_text = build_user_text(
        task="t2i", description="A photo of a corgi",
        aspect_ratio="1,1", resolution_w=960, resolution_h=960,
        version="v4.2-original",
    )

    # Text-to-image, anti-hallucination variant: adds source-anchoring +
    # rewrite-reflex suppression + person-attribute silence rules. Use
    # when the source caption is sparse and the upsampler must NOT invent
    # specifics absent from the source.
    user_text = build_user_text(
        task="t2i", description="A photo of a corgi",
        aspect_ratio="1,1", resolution_w=960, resolution_h=960,
        version="v4.2-constrained",
    )

    # Image-to-video (video parameters required)
    messages = build_messages(
        task="i2v", description="...",
        fps=20, duration_secs=20, aspect_ratio="9,16",
        resolution_w=480, resolution_h=832,
    )

    # Transfer-conditioned prompt upsampling (media supplied by caller)
    legacy_user_text = build_user_text(
        task="transfer", description="...",
        transfer_mode="v2v", control_modalities=["depth", "seg"],
    )
    structured_user_text = build_user_text(
        task="transfer", description="...",
        transfer_mode="v2v", control_modalities=["depth", "seg"],
        version="v4.2-structured",
    )

    # After receiving the model output:
    cleaned, _ = clean_response(raw_content, strip_think_when_appears=True)

    # Skip native upsampling on captions that already look upsampled
    # (e.g. fenced ``json`` payloads from a prior pass / external endpoint).
    if not is_upsampled_prompt(prompt):
        ...  # feed through ``upsample_captions``

Tasks: ``t2v`` (text-to-video), ``t2i`` (text-to-image), ``i2v`` (image-to-video),
``transfer`` (control-conditioned transfer prompt upsampling).
"""

import json as _json
import re as _re

SYSTEM_MESSAGE = "You are a helpful assistant."

# The primary dynamic field is ``{description}``: the source scene description
# the caller wants upsampled. For standard v4.2 tasks, all other content
# (instructions, task_constraints, output_json_template) stays fixed. Transfer
# additionally renders a mode-specific instruction using its control modalities.

_TEMPLATE_T2V_V4_2 = r"""<instructions>
You are a prompt upsampler for a text-to-video model. This instructions block defines the task and binds the output contract. The full input consists of this instructions block, a <video_description> scene description, a <task_constraints> numbered constraint list, and an <output_json_template> JSON schema. Produce exactly one fenced JSON object. The object fully populates every required field in the output template and strictly satisfies every numbered task constraint. The duration value specified in task constraint #4 is an upper bound for every timed field in the JSON, including the latest action end time and the final segment time_range; plan all timings within it.
</instructions>

<video_description>
{description}
</video_description>

<task_constraints>
1. **Scene imagination first.** Begin by writing `scene_imagination` first, before any other field, as ONE string containing ~6–12 short verb-led prompts (e.g., focus:, define:, refine:, visualize:, analyze:, clarify:). Keep it concise (prefer brevity over exhaustiveness) and under ~250 words. Use it to lock the concrete facts, style, camera intent, and timing plan; every other JSON field MUST be consistent with what you wrote here. (Operational note: at deployment, the inference team — not this upsampler — is responsible for stripping `scene_imagination` before the JSON is passed downstream.)

2. **Temporal caption second.** Immediately after `scene_imagination`, write `temporal_caption` as the second field. It is the canonical, timestamped visual playback timeline that downstream video generators consume to physicalize the clip. Walk the clip from t=0:00 to the `duration` (per #4) with timestamped beats placed where notable events occur (subject actions, camera motion, transitions, lighting/state changes, visual changes). Beat density must reflect how much actually happens: a busy clip will have many close-spaced beats; a slow contemplative clip will have a few sparse beats; a static shot may have only one beat at t=0:00 plus an ending observation. There is NO word limit — match the natural verbosity of the scene. Plan every other JSON field (subjects, actions, segments, etc.) to be consistent with the timeline you wrote here; in particular, `actions[].time` and `segments[].time_range` MUST agree with the beats you describe.

3. **Audio description third.** Immediately after `temporal_caption`, write `audio_description` as the third field. It is the parallel sonic timeline — dialogue, music, SFX, ambience — ideally aligned to the same beats with M:SS timestamps when audio events are time-localized (e.g. "soft ambient hum 0:00-0:03; sudden footstep at 0:04"). There is NO word limit — match the natural complexity of the scene's soundscape; if the clip is silent, write a brief description of why (e.g., "no audio track / silent footage").

4. **Output-parameter values (verbatim).** The following values are required to appear in the matching output JSON keys exactly as shown, and nowhere else; do not alter, normalize, round, infer, or relocate them:
   - duration: "{duration}"
   - fps: {fps}
   - aspect_ratio: "{aspect_ratio}"
   - resolution: {"W": {resolution_w}, "H": {resolution_h}}

5. **Duration is binding for time fields.** Treat the `duration` set in #4 as a hard ceiling for ALL timed content. Ensure `max(actions[].time.end)` ≤ `duration`, `max(segments[].time_range.end)` ≤ `duration`, and the latest beat in `temporal_caption` ≤ `duration`. Plan the schedule around this duration before filling actions/segments; after writing, re-check and truncate, shorten, or delete any item that would exceed the bound.

6. **Time format.** Use ONLY M:SS formatting for all time-bearing fields. `duration` must be exactly "M:SS" (e.g., "0:05"). Every `actions[].time` and `segments[].time_range` must be exactly "M:SS-M:SS" (e.g., "0:00-0:05"). Inside `temporal_caption` and `audio_description`, write timestamps in M:SS form (e.g., "At 0:03,…"). Never use "5s", never use milliseconds, and never use vague phase words like "beginning", "mid", "throughout".

7. **Internal consistency.** Make all fields mutually consistent and physically/visually coherent:
   - Maintain a logical temporal sequence across `temporal_caption`, `audio_description`, `actions`, and `segments`; no impossible overlaps unless explicitly intended and described.
   - Keep lighting, environment, wardrobe/props, mood, and color palette aligned across the whole JSON.
   - Ensure any lens/scale/perspective cues (e.g., wide vs telephoto, macro, aerial, POV) match stated framing, subject size, and placement.
   - Ensure `resolution.W`/`resolution.H` match the implied orientation and the (resolution_tier, aspect_ratio) dictated by #4.

8. **Faithfulness.** Do not contradict the provided video description. Every concrete described element (subjects, key actions, notable props, setting cues) must appear in the JSON, either directly or as a clearly plausible extension that does not change the core meaning.

9. **Scene coherence.** Add detail to increase richness, but only within the scene’s established logic and genre:
   - For realistic footage, obey physical plausibility (motion, lighting behavior, materials, causality).
   - For stylized/animated/sci-fi/fantasy/surreal scenes, prioritize internal stylistic rules and genre conventions over strict realism; additions must reinforce the chosen aesthetic.

10. **Schema completeness and density.** Output must strictly follow the template:
    - Include EVERY top-level key from the template exactly once; never invent extra keys.
    - Populate every field with specific, scene-justified detail; use empty values only when truly inapplicable.
    - The ONLY allowed empty values are exactly: `""`, `0`, `[]`, and `{}`. Do NOT use `null` anywhere.

11. **Output format.** Return ONLY a single JSON object wrapped inside a ```json code fence. Do not include prose, explanations, or any text outside the fence.

</task_constraints>

<output_json_template>
{
  "scene_imagination": "single string; verb-led scratchpad (focus:, define:, refine:, visualize:, analyze:); ~6-12 prompts; under ~250 words",
  "temporal_caption": "the canonical, timestamped playback timeline that downstream video generators consume to physicalize the clip; walk the clip from t=0:00 to the duration in #4 with timestamped beats placed where notable events occur (subject actions, camera motion, transitions, lighting/state changes, visual changes) — beat density should reflect how much actually happens (a busy clip will have many close beats; a slow contemplative clip will have a few sparse beats; a static shot may have only one beat at t=0:00 plus an ending observation); MUST be consistent with `actions[].time` and `segments[].time_range`; NO word limit — match the natural verbosity of the scene",
  "audio_description": "audio cues / soundscape for the clip — dialogue, music, SFX, ambience; ideally aligned to the beats in `temporal_caption` (use M:SS timestamps when audio events are time-localized, e.g. 'soft ambient hum 0:00-0:03; sudden footstep at 0:04')",

  "subjects": [
    {
      "description": "full visual description of the subject (appearance, clothing, identifying features)",
      "appearance_details": "additional visual details (accessories, distinguishing features)",
      "relationship": "how this subject relates to others or to the scene",
      "location": "where in frame (e.g., 'center foreground', 'left background')",
      "relative_size": "size within frame (e.g., 'Small within frame', 'Medium within frame', 'Large within frame')",
      "orientation": "direction subject faces relative to camera",
      "pose": "body position and posture",
      "action": "what the subject is doing (brief)",
      "state_changes": "how pose or action changes; 'No significant change.' if static",
      "clothing": "clothing and accessories; '' if non-human or not visible",
      "expression": "facial expression; '' if non-human or not visible",
      "gender": "one of 'Male', 'Female', 'Unknown'; '' if non-human",
      "age": "age category (e.g., 'Child', 'Young adult', 'Adult', 'Middle-aged', 'Elderly')",
      "skin_tone_and_texture": "skin tone description; '' if non-human",
      "facial_features": "notable facial features; '' if non-human or not visible",
      "number_of_subjects": "int; total in this subject's group; 0 if N/A",
      "number_of_arms": "int; 2 for humans, 0 if non-human",
      "number_of_legs": "int; 2 for humans, 0 if non-human"
    }
  ],
  "background_setting": "the setting / environment in which the scene takes place",
  "lighting": {
    "conditions": "overall lighting conditions (e.g., 'natural daylight', 'dim warm interior')",
    "direction": "primary light direction (e.g., 'front-lit', 'side-lit from left')",
    "shadows": "shadow character (e.g., 'soft', 'hard', 'long-cast')",
    "illumination_effect": "any notable illumination effect (e.g., 'rim-light', 'god rays', 'lens flare')"
  },
  "aesthetics": {
    "composition": "compositional choices (e.g., 'rule-of-thirds', 'symmetric', 'leading lines')",
    "color_scheme": "dominant color palette and mood",
    "mood_atmosphere": "emotional tone of the scene",
    "patterns": "notable repeating patterns; '' if none"
  },
  "cinematography": {
    "camera_motion": "camera motion (e.g., 'Static', 'Pan left', 'Tracking shot')",
    "framing": "shot framing (e.g., 'wide', 'medium', 'close-up')",
    "camera_angle": "camera angle (e.g., 'eye-level', 'high-angle', 'Dutch angle')",
    "depth_of_field": "depth-of-field choice (e.g., 'shallow', 'deep')",
    "focus": "what is in focus (e.g., 'subject in foreground; background bokeh')",
    "lens_focal_length": "focal length style (e.g., 'wide-angle 24mm', 'telephoto 85mm')"
  },
  "style_medium": "rendering style and medium (e.g., 'photoreal live-action', 'cel-shaded animation')",
  "artistic_style": "broader artistic style if applicable (e.g., 'noir', 'pastoral painterly')",
  "context": "broader narrative or situational context",
  "actions": [
    {
      "time": "M:SS-M:SS time range (e.g., '0:00-0:03'); end MUST be <= duration",
      "description": "what happens during this time range"
    }
  ],
  "text_and_signage_elements": [
    {
      "text": "the exact text/sign content",
      "category": "one of 'physical_in_scene', 'scene_sign', 'ui_text', 'caption', 'logo', or similar",
      "appearance": "how the text appears (font style, color, size)",
      "spatial_temporal": "where and when in the clip the text appears",
      "context": "narrative or situational context for the text"
    }
  ],
  "segments": [
    {
      "segment_index": "int; 0-based ordinal of this segment",
      "time_range": "M:SS-M:SS time range; end MUST be <= duration",
      "description": "what happens in this segment",
      "key_changes": "key state/scene changes within this segment",
      "camera": "camera behaviour within this segment"
    }
  ],
  "transitions": [
    "transition descriptions between segments; [] if single continuous shot"
  ],

  "resolution":   "Per task constraint #4",
  "aspect_ratio": "Per task constraint #4",
  "duration":     "Per task constraint #4",
  "fps":          "Per task constraint #4"
}

</output_json_template>"""

_TEMPLATE_T2I_V4_2 = r"""<instructions>
You are a prompt upsampler for a text-to-image model. This instructions block governs the response. Next come an <image_description> scene description, a <task_constraints> numbered constraint list, and an <output_json_template> JSON schema. Produce exactly one fenced JSON object. The object fully populates every field in the template, uses valid JSON, and strictly satisfies each numbered task constraint without omission or deviation.
</instructions>

<image_description>
{description}
</image_description>

<task_constraints>
1. **Scene imagination first.** Begin by filling `scene_imagination` first, before any other field, as one single string made of short verb-led prompts (e.g., focus:, define:, refine:, visualize:, analyze:). Write ~6–12 prompts, prioritize brevity over exhaustiveness, and keep the total under ~250 words. Ensure every later field is consistent with what you wrote here. (Operational note: at deployment, the inference team — not this upsampler — is responsible for stripping `scene_imagination` before the JSON is passed downstream.)

2. **Comprehensive T2I caption — pinned 2nd, dense, downstream-actionable.** After `scene_imagination`, populate `comprehensive_t2i_caption` immediately (it MUST remain the 2nd top-level key in the output JSON). This text is the prompt consumed by the downstream image generator; all other JSON keys exist to support it.

- **Density**: 80–200 words, composed as a SINGLE tight paragraph (1–3 sentences). Not a one-line synopsis; not a list.

- **Integration**: merge EVERY concrete visual detail from the rest of the JSON — primary and secondary subjects (appearance, wardrobe, facial expression, pose), background_setting, lighting (conditions, direction, shadow behavior, illumination effects), aesthetics (composition, color palette, mood, patterns), cinematography (framing, camera angle, depth-of-field, focus, lens style), style_medium, artistic_style, plus any legible text/signage. Do not exclude any concrete item present in `subjects[]` or other populated fields.

- **Phrasing**: be immediate and literal. Start with the subject in the setting — NEVER begin with “this image shows,” “an image of,” “a picture of,” “depicting,” “we see,” or any meta-intro framing it as an image description.  
    DO  (good — begins with subject):  
        "A young woman in a crimson dress stands at the rim of a moonlit canyon..."  
    DON'T  (bad — meta-prefaced):  
        "This image shows a young woman in a crimson dress standing at the rim..."

- **Specificity**: use only visually-executable adjectives. Swap vague terms (“good lighting,” “nice composition”) for concrete directives (“warm late-afternoon golden light raking across...,” “rule-of-thirds with the subject anchored bottom-left”).

- **Consistency**: every stated element MUST be consistent with `subjects[]`, `background_setting`, `lighting`, `aesthetics`, `cinematography`, `style_medium`, `artistic_style`, and `text_and_signage_elements`. Do not add ungrounded details, and do not omit concrete details that are specified there.

3. **Output-parameter copy.** Copy these exact values into the matching output JSON keys, byte-for-byte:
   - aspect_ratio: "{aspect_ratio}"
   - resolution: {"W": {resolution_w}, "H": {resolution_h}}
   Do not modify, normalize, round, infer alternates, or relocate them. (T2I has no duration or fps — those are video-only.)

4. **Internal consistency.**
   - Keep lighting, setting, time-of-day, camera/framing, and mood mutually consistent (no contradictions such as “harsh noon sun” with “dim candlelit interior” unless explicitly justified).
   - Ensure `resolution.W` and `resolution.H` exactly match the (resolution_tier, aspect_ratio) pair implied by constraint #3; do not invent other sizes.

5. **Faithfulness.** Do not contradict the provided image description. Every concrete element mentioned (subjects, actions/poses, wardrobe, props, background features, environment, style cues) must appear in the JSON or be extended only in a way that is clearly plausible and non-conflicting.

6. **Scene coherence.** Add only details that fit the scene’s established logic and style:
   - For realistic scenes, additions should be physically plausible and context-appropriate.
   - For animation/sci‑fi/fantasy/surreal or other non-realistic genres, additions must follow that genre’s conventions and maintain a consistent visual language (do not “force realism” into a stylized world).

7. **Schema completeness and density.**
   - Include every top-level key from the template, and never add keys beyond the template.
   - Populate every field with specific, accurate, image-grounded detail; treat empties as a last resort.
   - The only permitted empty values are exactly: `""`, `0`, `[]`, `{}`. Do not use `null` anywhere.
   - If there is any human or humanoid subject, set `number_of_hands = 2` and `number_of_fingers = 10`; if all subjects are non-human, set both to 0.

8. **subject_details density (T2I-only).** Ensure the top-level `subject_details` dict is present and non-empty. Choose 2–5 image-specific attribute keys that best characterize THIS image, with concrete descriptive string values; vary the keys per image. Never output `{}`. Do not reuse any `subjects[].*` field names as `subject_details` keys (those belong inside per-subject objects).

9. **Output format.** Return ONLY the single JSON object, wrapped inside a ```json code fence. Do not include prose, explanations, comments, or any text outside the fence.
</task_constraints>

<output_json_template>
{
  "scene_imagination": "single string; verb-led scratchpad (focus:, define:, refine:, visualize:, analyze:); ~6-12 prompts; under ~250 words",
  "comprehensive_t2i_caption": "Per task constraint #2",
  "subjects": [
    {
      "description": "full visual description of the subject",
      "appearance_details": "additional visual details (accessories, texture, distinguishing features)",
      "relationship": "how this subject relates to others or to the scene",
      "location": "where in frame (e.g., 'Center foreground', 'Top right')",
      "relative_size": "size within frame",
      "orientation": "direction subject faces relative to camera",
      "pose": "body position and posture",
      "clothing": "clothing and accessories; '' if non-human or N/A",
      "expression": "facial expression; '' if non-human or N/A",
      "gender": "one of 'Male', 'Female', 'Unknown', 'N/A'",
      "age": "age category",
      "skin_tone_and_texture": "skin tone description; '' if non-human",
      "facial_features": "notable facial features; '' if non-human or not visible",
      "number_of_subjects": "int; total in this subject's group; 0 if N/A",
      "number_of_arms": "int; 2 for humans, 0 if non-human",
      "number_of_legs": "int; 2 for humans, 0 if non-human",
      "number_of_hands": "int; 2 for humans, 0 if non-human",
      "number_of_fingers": "int; 10 for humans, 0 if non-human"
    }
  ],
  "subject_details": {
    "<key_name>": "free-form image-specific structured attribute; keys vary per image; {} if N/A"
  },
  "background_setting": "the setting / environment in which the image is set",
  "lighting": {
    "conditions": "overall lighting conditions",
    "direction": "primary light direction",
    "shadows": "shadow character",
    "illumination_effect": "any notable illumination effect"
  },
  "aesthetics": {
    "composition": "compositional choices",
    "color_scheme": "dominant color palette and mood",
    "mood_atmosphere": "emotional tone of the image",
    "patterns": "notable repeating patterns; '' if none"
  },
  "cinematography": {
    "framing": "shot framing (e.g., 'wide', 'medium', 'close-up')",
    "camera_angle": "camera angle (e.g., 'eye-level', 'high-angle', 'Dutch angle')",
    "depth_of_field": "depth-of-field choice (e.g., 'shallow', 'deep')",
    "focus": "what is in focus",
    "lens_focal_length": "focal length style (e.g., 'wide-angle 24mm', 'telephoto 85mm')"
  },
  "style_medium": "rendering style and medium (e.g., 'photoreal photograph', 'oil-painting style')",
  "artistic_style": "broader artistic style if applicable",
  "context": "broader narrative or situational context",
  "text_and_signage_elements": [
    {
      "text": "the exact text/sign content",
      "category": "one of 'physical_in_scene', 'scene_sign', 'ui_text', 'caption', 'logo', or similar",
      "appearance": "how the text appears (font style, color, size)",
      "spatial": "where in the image the text appears",
      "context": "narrative or situational context for the text"
    }
  ],
  "quadrant_scan": {
    "top_left": "what is in the top-left region",
    "top_right": "what is in the top-right region",
    "bottom_left": "what is in the bottom-left region",
    "bottom_right": "what is in the bottom-right region",
    "absolute_center": "what is in the dead-center of the frame"
  },
  "resolution": "Per task constraint #3",
  "aspect_ratio": "Per task constraint #3"
}
</output_json_template>"""

_TEMPLATE_T2I_V4_2_CONSTRAINED = r"""<instructions>
You are a prompt upsampler for a text-to-image model. This instructions block governs the response. Next come an <image_description> scene description, a <task_constraints> numbered constraint list, and an <output_json_template> JSON schema. Produce exactly one fenced JSON object. The object fully populates every field in the template, uses valid JSON, and strictly satisfies each numbered task constraint without omission or deviation.
</instructions>

<image_description>
{description}
</image_description>

<task_constraints>
1. **SOURCE ANCHORING (highest priority).** Every concrete noun anywhere in your output — entity, material, color, count, named object, person attribute (gender, age, wardrobe, hair, skin, facial feature), on-screen or signage text, displayed number, brand, label — MUST already appear in <image_description>, either verbatim or as a direct unambiguous paraphrase. If <image_description> is silent on a property, your output must also be silent: leave the field empty or use the generic word the source used.

   The rule is categorical, not phrase-specific. Apply it to any input by recognizing the category. Pattern templates:
   - If source names an OBJECT generically (e.g. "the device", "a tool") without specifying material → do not invent a material in your output.
   - If source names a COUNT generically ("dozens", "several", "many") → keep the same generic word; do not pick a specific number ("over fifty", "twelve").
   - If source mentions a PERSON without an attribute (gender / age / hair / skin / wardrobe) → do not introduce that attribute. Use the same generic word the source used.
   - If source mentions a SIGN or SCREEN-TEXT without giving the exact text content → do not invent text content; describe the sign generically.
   - If source uses a precise PART-NAME or named component → reuse that exact phrase wherever you mention the thing.
   These templates illustrate the principle; the principle applies to any noun in any input.

2. **REWRITE-REFLEX SUPPRESSION.** Do NOT rewrite a source-anchored phrase into a different specific. If <image_description> already states a concrete noun, copy that noun verbatim wherever you mention the thing. Do NOT substitute a synonym, do NOT category-shift (e.g. "X" → "X-variant"), do NOT pick a more-specific subtype, do NOT replace the source phrase with an "elaborated" equivalent. Photographic descriptors (lens, lighting, framing) are NOT concrete nouns and may be added freely.

3. **CHAIN-LOCK across fields.** Every concrete noun in `comprehensive_t2i_caption`, in any `subjects[]` field, in `background_setting`, in `subject_details`, in `quadrant_scan`, in `text_and_signage_elements`, and in `context` MUST already appear in `scene_imagination` OR in <image_description>. New concrete nouns may NOT be introduced after scene_imagination is written. The only freely-added content across all fields is the photographic-descriptor class (see constraint #5).

4. **PERSON-ATTRIBUTE SILENCE (HARD RULE for human / humanoid subjects).** When <image_description> mentions a person without specifying an attribute, use these defaults — never invent:
   - `gender` = "Unknown" unless <image_description> uses an explicit gendered word
   - `age` = the exact age word <image_description> uses (or "Unknown" if no age word)
   - `clothing` = "" unless <image_description> mentions clothing
   - `skin_tone_and_texture` = "" unless <image_description> mentions skin
   - `facial_features` = "" unless <image_description> mentions a specific feature
   - `expression` = the exact word <image_description> uses (or "" if no expression word)
   This rule applies to every human/humanoid subject in `subjects[]`. Filling these slots from your training prior is forbidden when source is silent.

5. **PHOTOGRAPHIC DESCRIPTORS — freely add.** Only the following classes may be invented (because they describe HOW the image is captured, not WHAT is in it):
   - camera framing / angle / lens / focal length / depth of field
   - lighting quality / direction (only when source mentions or implies lighting)
   - composition (rule-of-thirds, leading lines, symmetry, negative space)
   - rendering style (photoreal / illustration / cartoon — only when source implies)
   - atmospheric quality (haze, contrast, mood, color palette)
   - generic shadow / reflection / specular behavior

6. **Order of generation.** First fill `scene_imagination` (verb-led scratchpad, 6-12 prompts, ~250 words max) using source-anchored vocabulary. Then fill `comprehensive_t2i_caption` (one tight paragraph, 80-200 words) reusing scene_imagination's concrete vocabulary verbatim — only adding photographic descriptors. Then fill the remaining structured fields (subjects[], background_setting, etc.), all of which inherit vocabulary from constraint #3.

7. **Output-parameter copy.** Copy these values byte-for-byte:
   - aspect_ratio: "{aspect_ratio}"
   - resolution: {"W": {resolution_w}, "H": {resolution_h}}

8. **Internal consistency.** Lighting / setting / time-of-day / framing / mood must be mutually consistent.

9. **Schema completeness.** Include every top-level key from the template; never add keys; never omit keys. Permitted empties: "", 0, [], {}. No null. If any subject is human/humanoid: number_of_hands=2, number_of_fingers=10. If all non-human: both = 0.

10. **subject_details density.** `subject_details` non-empty with 2-5 source-anchored attribute keys. Never `{}`.

11. **Output format.** ONLY one JSON object inside a ```json code fence. No prose outside.
</task_constraints>

<output_json_template>
{
  "scene_imagination": "Per #1, #2, #6 — verb-led scratchpad with source-anchored concrete nouns only; under ~250 words",
  "comprehensive_t2i_caption": "Per #3, #6 — same concrete vocabulary as scene_imagination plus photographic descriptors only",
  "subjects": [
    {
      "description": "Per #1, #3 — source-anchored only",
      "appearance_details": "Per #1, #3 — source-anchored only",
      "relationship": "how this subject relates to others",
      "location": "where in frame",
      "relative_size": "size within frame",
      "orientation": "direction subject faces relative to camera",
      "pose": "body position and posture",
      "clothing": "Per #4 — '' if source-silent or non-human",
      "expression": "Per #4 — '' if source-silent or non-human",
      "gender": "Per #4 — 'Unknown' if source-silent",
      "age": "Per #4 — source's word verbatim; 'Unknown' if no age word",
      "skin_tone_and_texture": "Per #4 — '' if source-silent or non-human",
      "facial_features": "Per #4 — '' if source-silent or non-human",
      "number_of_subjects": "int; total in this subject's group; 0 if N/A",
      "number_of_arms": "int",
      "number_of_legs": "int",
      "number_of_hands": "int",
      "number_of_fingers": "int"
    }
  ],
  "subject_details": { "<key_name>": "Per #1, #3 — source-anchored attribute" },
  "background_setting": "Per #1, #3 — source-anchored",
  "lighting": { "conditions": "...", "direction": "...", "shadows": "...", "illumination_effect": "..." },
  "aesthetics": { "composition": "...", "color_scheme": "...", "mood_atmosphere": "...", "patterns": "" },
  "cinematography": { "framing": "...", "camera_angle": "...", "depth_of_field": "...", "focus": "...", "lens_focal_length": "..." },
  "style_medium": "rendering style per source",
  "artistic_style": "broader style only if source implies",
  "context": "Per #1, #3 — source-anchored",
  "text_and_signage_elements": [
    { "text": "exact source text; entry omitted if source-silent on specific text content", "category": "...", "appearance": "...", "spatial": "...", "context": "..." }
  ],
  "quadrant_scan": { "top_left": "Per #1, #3", "top_right": "Per #1, #3", "bottom_left": "Per #1, #3", "bottom_right": "Per #1, #3", "absolute_center": "Per #1, #3" },
  "resolution": "Per #7",
  "aspect_ratio": "Per #7"
}
</output_json_template>"""


_TEMPLATE_T2I_V4_2_EXPRESSIVE = r"""<instructions>
You are a prompt upsampler for a text-to-image model. Your job is to UPSAMPLE — take a sparse natural-language request and expand it into a rich, dense, structured JSON description of the target image. This instructions block governs the response. Next come an <image_description> scene description, a <task_constraints> numbered constraint list, and an <output_json_template> JSON schema. Produce exactly one fenced JSON object that fully populates every top-level key, satisfies every numbered task constraint, and is internally consistent with the request.

The output is always DENSE. Even when the request is brief, infer plausible, scene-consistent details for every field. Do not leave fields empty merely because the request did not mention them — the purpose of upsampling is to turn a sparse request into a complete, image-ready annotation. Be creative but stay grounded: additions must be physically plausible and internally consistent with the request's setting, subjects, mood, and context.
</instructions>

<image_description>
{description}
</image_description>

<task_constraints>
1. **Scene imagination first.** Begin by filling `scene_imagination` first, before any other field, as one single string made of short verb-led prompts (e.g., focus:, define:, refine:, visualize:, analyze:). Write ~6-12 prompts, under ~250 words total. Use this as your scratchpad for the whole scene: focus the main subject, define key elements, refine details, visualize lighting/camera/atmosphere, analyze coherence. Every later field must be consistent with what you wrote here. (Operational note: at deployment, the inference team strips `scene_imagination` before the JSON is passed downstream.)

2. **Comprehensive T2I caption — pinned 2nd, dense, downstream-actionable.** After `scene_imagination`, populate `comprehensive_t2i_caption` immediately (MUST remain the 2nd top-level key). This is the natural-language prose passed to the downstream image generator; all other JSON keys exist to support it.

   - **Density**: 80-200 words as a SINGLE tight paragraph (1-3 sentences). Not a one-line synopsis; not a list.
   - **Integration**: merge EVERY concrete detail from the structured fields you populate below — primary and secondary subjects (appearance, wardrobe, expression, pose), background_setting, lighting (conditions, direction, shadow behavior, illumination effect), aesthetics (composition, palette, mood, patterns), cinematography (framing, angle, depth-of-field, focus, lens), style_medium, artistic_style, and any visible text/signage. Do not exclude any concrete item present in `subjects[]` or other populated fields.
   - **Phrasing**: be immediate and literal. Start with the subject in the setting — NEVER begin with "this image shows", "an image of", "a picture of", "depicting", "we see", or any meta-intro.
     DO  : "A young woman in a crimson dress stands at the rim of a moonlit canyon..."
     DON'T: "This image shows a young woman in a crimson dress standing at the rim..."
   - **Specificity**: use visually-executable adjectives. Swap vague terms ("good lighting") for concrete directives ("warm late-afternoon golden light raking across...").

3. **Output-parameter copy.** Copy these exact values into the matching output JSON keys, byte-for-byte:
   - aspect_ratio: "{aspect_ratio}"
   - resolution: {"W": {resolution_w}, "H": {resolution_h}}
   Do not modify, normalize, or relocate them. (T2I has no duration or fps.)

4. **Internal consistency.** Lighting / setting / time-of-day / camera / framing / mood must be mutually consistent. No contradictions (e.g. "harsh noon sun" with "dim candlelit interior" unless justified).

5. **Faithfulness.** Do not contradict the provided image description. Every concrete element it mentions (subjects, actions/poses, wardrobe, props, background features, environment, style cues) must appear in the JSON or be extended in a way that is clearly plausible and non-conflicting.

6. **EXPRESSIVE DENSITY (highest priority for empties).** The purpose of the upsampler is to FILL the structured annotation, not echo it. Even when the request is brief, infer plausible details for every field consistent with the request's scene, subjects, and mood. Be creative but stay grounded:
   - Additions must be physically plausible and internally consistent.
   - For realistic scenes, additions are physically plausible and context-appropriate.
   - For animation/sci-fi/fantasy/surreal, additions follow that genre's conventions and visual language.
   - Inferences must support the comprehensive_t2i_caption — not contradict source, not introduce conflicting elements.

7. **Schema completeness and permitted empties.** Include every top-level key from the template exactly once. Never add keys beyond the template. Populate every field with specific, image-grounded detail. Empty values are permitted ONLY for truly inapplicable fields:
   - Human-only subject fields (clothing, expression, gender, age, skin_tone_and_texture, facial_features, number_of_arms, number_of_legs, number_of_hands, number_of_fingers) when the subject is non-human.
   - `text_and_signage_elements = []` when no visible text or signage is present.
   - `aesthetics.patterns = ""` when there are no notable repeating patterns.
   - `subject_details = {}` when no image-specific structured attributes apply.
   The only permitted empty literals are exactly: `""`, `0`, `[]`, `{}`. Do not use `null`.
   - If any subject is human/humanoid: set `number_of_hands = 2` and `number_of_fingers = 10`. If all subjects are non-human, set both to 0.

8. **subject_details density (T2I-only).** Top-level `subject_details` dict is present and non-empty: 2-5 image-specific attribute keys with concrete descriptive string values (e.g. `"hairstyle": "wavy auburn shoulder-length"`, `"footwear": "tan leather Chelsea boots"`, `"hand_props": "antique brass pocket watch in right hand"`). Vary keys per image; never reuse `subjects[].*` field names; never output `{}` when at least one human/humanoid subject is present.

9. **Output format.** Return ONLY the single JSON object, wrapped inside a ```json code fence. No prose, explanations, comments, or text outside the fence.
</task_constraints>

<output_json_template>
{
  "scene_imagination": "single string; verb-led scratchpad (focus:, define:, refine:, visualize:, analyze:); ~6-12 prompts; under ~250 words",
  "comprehensive_t2i_caption": "Per task constraint #2 — dense single paragraph, 80-200 words, integrates every concrete item from structured fields, starts with subject-in-setting, no meta intros",
  "subjects": [
    {
      "description": "full visual description of the subject (appearance, identifying features, distinctive traits)",
      "appearance_details": "secondary visual details (accessories, textures, surface character)",
      "relationship": "how this subject relates to others or to the scene",
      "location": "where in frame (e.g., 'Center foreground', 'Top right')",
      "relative_size": "size within frame (e.g., 'Small within frame', 'Medium within frame', 'Large within frame')",
      "orientation": "direction subject faces relative to camera",
      "pose": "body position and posture",
      "clothing": "clothing and accessories; '' if non-human",
      "expression": "facial expression; '' if non-human or not visible",
      "gender": "one of 'Male', 'Female', 'Unknown', 'N/A'",
      "age": "age category (e.g., 'Child', 'Young adult', 'Adult', 'Middle-aged', 'Elderly')",
      "skin_tone_and_texture": "skin tone and texture description; '' if non-human",
      "facial_features": "notable facial features incl. eye shape/color, hair color/style/length, lip shape, wrinkles, moles, scars, freckles, facial hair, glasses, makeup, and other visible fine-grained facial attributes; '' if non-human or not visible",
      "number_of_subjects": "int; total in this subject's group; 0 if N/A",
      "number_of_arms": "int; 2 for humans, 0 if non-human",
      "number_of_legs": "int; 2 for humans, 0 if non-human",
      "number_of_hands": "int; 2 for humans, 0 if non-human",
      "number_of_fingers": "int; 10 for humans, 0 if non-human"
    }
  ],
  "subject_details": {
    "<key_name>": "free-form image-specific structured attribute; keys vary per image; '' value strings allowed but never the whole dict empty"
  },
  "background_setting": "full prose description of the environment / setting / context behind the main subject(s)",
  "lighting": {
    "conditions": "type and quality of light (e.g., 'Bright daylight', 'Overcast', 'Studio lighting', 'Golden hour')",
    "direction": "primary light direction (e.g., 'top-lit', 'front-lit', 'side-lit from right')",
    "shadows": "shadow character (e.g., 'soft', 'hard', 'long-cast')",
    "illumination_effect": "any notable illumination effect (e.g., 'rim-light', 'god rays', 'lens flare', 'soft fill')"
  },
  "aesthetics": {
    "composition": "compositional choices (e.g., 'rule-of-thirds', 'symmetric', 'leading lines', 'center-weighted')",
    "color_scheme": "dominant color palette and mood",
    "mood_atmosphere": "emotional tone of the image",
    "patterns": "notable repeating visual patterns; '' if none"
  },
  "cinematography": {
    "framing": "shot framing (e.g., 'wide', 'medium', 'close-up')",
    "camera_angle": "camera angle (e.g., 'eye-level', 'high-angle', 'Dutch angle')",
    "depth_of_field": "depth-of-field choice (e.g., 'shallow', 'deep', 'uniform focus')",
    "focus": "what is in sharp focus (e.g., 'subject in foreground; background bokeh')",
    "lens_focal_length": "focal length style (e.g., 'wide-angle 24mm', 'telephoto 85mm')"
  },
  "style_medium": "rendering style and medium (e.g., 'photoreal photograph', 'oil painting', 'cel-shaded animation', 'digital presentation slide', 'screenshot')",
  "artistic_style": "broader artistic style if applicable (e.g., 'noir', 'pastoral painterly', 'cyberpunk')",
  "context": "broader narrative or situational context (brief)",
  "text_and_signage_elements": [
    {
      "text": "the exact text/sign content",
      "category": "one of 'physical_in_scene', 'scene_sign', 'ui_text', 'body_text', 'caption', 'logo', 'label'",
      "appearance": "how the text appears (font style, color, size, weight)",
      "spatial": "where in the image the text appears",
      "context": "narrative or situational context for the text"
    }
  ],
  "quadrant_scan": {
    "top_left": "what is in the top-left region",
    "top_right": "what is in the top-right region",
    "bottom_left": "what is in the bottom-left region",
    "bottom_right": "what is in the bottom-right region",
    "absolute_center": "what is in the dead-center of the frame"
  },
  "resolution": "Per task constraint #3",
  "aspect_ratio": "Per task constraint #3"
}
</output_json_template>"""


_TEMPLATE_I2V_V4_2 = r"""<instructions>
Your function is to operate as a prompt upsampler for an image-to-video model. You will be provided with several inputs: (a) an attached starting frame image, which serves as the definitive visual ground truth for subjects, setting, lighting, and color palette; (b) this instruction block; (c) a <video_description> detailing the scene's temporal and action-based intent; (d) a numbered <task_constraints> list; and (e) an <output_json_template> schema. Your sole output is one fenced JSON object. This object must populate every required field from the template and meticulously satisfy every numbered task constraint. Fields pertaining to visual information (`subjects`, `background_setting`, `lighting`, `aesthetics`, `style_medium`, `artistic_style`) must be entirely consistent with the attached image and must not contradict it. Fields pertaining to temporal information (`actions`, `segments`, `transitions`, `temporal_caption`) should be derived from the <video_description>, allowing for plausible extrapolation of events beyond the static first frame. The duration value from task constraint #2 establishes a strict upper limit for all time-based values in the JSON, which includes the latest action end time and the closing `time_range` of the final segment; all scheduling must occur within this duration.

</instructions>

<video_description>
{description}
</video_description>

<task_constraints>
1. **Scene imagination first.** Begin by writing `scene_imagination` first, before any other field, as ONE string containing ~6–12 short verb-led prompts (e.g., focus:, define:, refine:, visualize:, analyze:, clarify:). Keep it concise (prefer brevity over exhaustiveness) and under ~250 words. Use it to lock the concrete facts, style, camera intent, and timing plan; every other JSON field MUST be consistent with what you wrote here. Anchor the visual facts (subjects' appearance, setting, lighting, palette) on the attached starting frame; anchor the temporal facts (action sequence, segment timeline, transitions) on the video_description. (Operational note: at deployment, the inference team — not this upsampler — is responsible for stripping `scene_imagination` before the JSON is passed downstream.)

2. **Output-parameter copy.** Copy these exact values into the matching output JSON keys and nowhere else; do not alter, normalize, round, infer, or relocate them:
   - duration: "{duration}"
   - fps: {fps}
   - aspect_ratio: "{aspect_ratio}"
   - resolution: {"W": {resolution_w}, "H": {resolution_h}}

3. **Duration is binding for time fields.** Treat the `duration` copied in #2 as a hard ceiling for ALL timed content. Ensure `max(actions[].time.end)` ≤ `duration` and `max(segments[].time_range.end)` ≤ `duration`. Plan the schedule around this duration before filling actions/segments; after writing, re-check and truncate, shorten, or delete any item that would exceed the bound.

4. **Time format.** Use ONLY M:SS formatting for all time-bearing fields. `duration` must be exactly "M:SS" (e.g., "0:05"). Every `actions[].time` and `segments[].time_range` must be exactly "M:SS-M:SS" (e.g., "0:00-0:05"). Never use "5s", never use milliseconds, and never use vague phase words like "beginning", "mid", "throughout".

5. **Internal consistency.** Make all fields mutually consistent and physically/visually coherent:
   - Maintain a logical temporal sequence across actions and segments; no impossible overlaps unless explicitly intended and described.
   - Keep lighting, environment, wardrobe/props, mood, and color palette aligned across the whole JSON, anchored on what is visible in the attached starting frame.
   - Ensure any lens/scale/perspective cues (e.g., wide vs telephoto, macro, aerial, POV) match stated framing, subject size, and placement, consistent with the attached frame.
   - Ensure `resolution.W`/`resolution.H` match the implied orientation and the (resolution_tier, aspect_ratio) dictated by #2.

6. **Faithfulness to BOTH image and description.** Do not contradict either input. Every concrete element visible in the attached starting frame (subjects, their appearance, setting, lighting, colors, framing) MUST appear in the JSON. Every concrete described element in the video_description (key actions, notable temporal beats, mentioned props) MUST also appear. When the description is ambiguous on a visual field, defer to the image; when the image cannot answer a temporal question, defer to the description.

7. **Visual fidelity to the attached frame.** The attached image is the FIRST FRAME of the video; the video starts here. The first segment (`segments[0]`) and the earliest actions must depict the scene as shown in the image — same subjects in the same configuration, same setting, same lighting, same color palette. Subsequent segments may evolve naturally; the scene at t=0 must match the image.

8. **Scene coherence.** Add detail to increase richness, but only within the scene's established logic and genre:
   - For realistic footage, obey physical plausibility (motion, lighting behavior, materials, causality).
   - For stylized/animated/sci-fi/fantasy/surreal scenes, prioritize internal stylistic rules and genre conventions over strict realism; additions must reinforce the chosen aesthetic established by the image.

9. **Schema completeness and density.** Output must strictly follow the template:
   - Include EVERY top-level key from the template exactly once; never invent extra keys.
   - Populate every field with specific, scene-justified detail; use empty values only when truly inapplicable.
   - The ONLY allowed empty values are exactly: `""`, `0`, `[]`, and `{}`. Do NOT use `null` anywhere.

10. **Output format.** Return ONLY a single JSON object wrapped inside a ```json code fence. Do not include prose, explanations, or any text outside the fence.

</task_constraints>

<output_json_template>
{
  "scene_imagination": "single string; verb-led scratchpad (focus:, define:, refine:, visualize:, analyze:); ~6-12 prompts; under ~250 words",
  "temporal_caption": "single sentence describing the temporal arc of the clip",
  "audio_description": "audio cues / soundscape for the clip",

  "subjects": [
    {
      "description": "full visual description of the subject (appearance, clothing, identifying features)",
      "appearance_details": "additional visual details (accessories, distinguishing features)",
      "relationship": "how this subject relates to others or to the scene",
      "location": "where in frame (e.g., 'center foreground', 'left background')",
      "relative_size": "size within frame (e.g., 'Small within frame', 'Medium within frame', 'Large within frame')",
      "orientation": "direction subject faces relative to camera",
      "pose": "body position and posture",
      "action": "what the subject is doing (brief)",
      "state_changes": "how pose or action changes; 'No significant change.' if static",
      "clothing": "clothing and accessories; '' if non-human or not visible",
      "expression": "facial expression; '' if non-human or not visible",
      "gender": "one of 'Male', 'Female', 'Unknown'; '' if non-human",
      "age": "age category (e.g., 'Child', 'Young adult', 'Adult', 'Middle-aged', 'Elderly')",
      "skin_tone_and_texture": "skin tone description; '' if non-human",
      "facial_features": "notable facial features; '' if non-human or not visible",
      "number_of_subjects": "int; total in this subject's group; 0 if N/A",
      "number_of_arms": "int; 2 for humans, 0 if non-human",
      "number_of_legs": "int; 2 for humans, 0 if non-human"
    }
  ],
  "background_setting": "the setting / environment in which the scene takes place",
  "lighting": {
    "conditions": "overall lighting conditions (e.g., 'natural daylight', 'dim warm interior')",
    "direction": "primary light direction (e.g., 'front-lit', 'side-lit from left')",
    "shadows": "shadow character (e.g., 'soft', 'hard', 'long-cast')",
    "illumination_effect": "any notable illumination effect (e.g., 'rim-light', 'god rays', 'lens flare')"
  },
  "aesthetics": {
    "composition": "compositional choices (e.g., 'rule-of-thirds', 'symmetric', 'leading lines')",
    "color_scheme": "dominant color palette and mood",
    "mood_atmosphere": "emotional tone of the scene",
    "patterns": "notable repeating patterns; '' if none"
  },
  "cinematography": {
    "camera_motion": "camera motion (e.g., 'Static', 'Pan left', 'Tracking shot')",
    "framing": "shot framing (e.g., 'wide', 'medium', 'close-up')",
    "camera_angle": "camera angle (e.g., 'eye-level', 'high-angle', 'Dutch angle')",
    "depth_of_field": "depth-of-field choice (e.g., 'shallow', 'deep')",
    "focus": "what is in focus (e.g., 'subject in foreground; background bokeh')",
    "lens_focal_length": "focal length style (e.g., 'wide-angle 24mm', 'telephoto 85mm')"
  },
  "style_medium": "rendering style and medium (e.g., 'photoreal live-action', 'cel-shaded animation')",
  "artistic_style": "broader artistic style if applicable (e.g., 'noir', 'pastoral painterly')",
  "context": "broader narrative or situational context",
  "actions": [
    {
      "time": "M:SS-M:SS time range (e.g., '0:00-0:03'); end MUST be <= duration",
      "description": "what happens during this time range"
    }
  ],
  "text_and_signage_elements": [
    {
      "text": "the exact text/sign content",
      "category": "one of 'physical_in_scene', 'scene_sign', 'ui_text', 'caption', 'logo', or similar",
      "appearance": "how the text appears (font style, color, size)",
      "spatial_temporal": "where and when in the clip the text appears",
      "context": "narrative or situational context for the text"
    }
  ],
  "segments": [
    {
      "segment_index": "int; 0-based ordinal of this segment",
      "time_range": "M:SS-M:SS time range; end MUST be <= duration",
      "description": "what happens in this segment",
      "key_changes": "key state/scene changes within this segment",
      "camera": "camera behaviour within this segment"
    }
  ],
  "transitions": [
    "transition descriptions between segments; [] if single continuous shot"
  ],

  "resolution":   "Per task constraint #2",
  "aspect_ratio": "Per task constraint #2",
  "duration":     "Per task constraint #2",
  "fps":          "Per task constraint #2"
}
</output_json_template>"""


_TEMPLATE_TRANSFER_OUTPUT_JSON_V4_2 = r"""{
  "subjects": [
    {
      "description": "full visual description",
      "appearance_details": "clothing, accessories, distinguishing features",
      "relationship": "how subject interacts with other elements",
      "location": "where in frame (e.g. center foreground, left background)",
      "relative_size": "size within frame (e.g. large, small, medium)",
      "orientation": "facing direction relative to camera",
      "pose": "body position and posture",
      "action": "what the subject is doing",
      "state_changes": "how pose or action changes over time"
    }
  ],
  "background_setting": "full description of environment and setting",
  "lighting": {
    "conditions": "type of light (e.g. overcast daylight, studio, golden hour)",
    "direction": "where light comes from (e.g. side-lit from right, front-lit)",
    "shadows": "shadow description and which side they fall on",
    "illumination_effect": "overall effect on the scene"
  },
  "aesthetics": {
    "composition": "framing and compositional choices",
    "color_scheme": "dominant colors and tones",
    "mood_atmosphere": "emotional atmosphere"
  },
  "cinematography": {
    "framing": "shot type (e.g. close-up, medium, wide)",
    "camera_motion": "camera movement (e.g. pan left, static, tracking shot)",
    "camera_angle": "angle (e.g. eye-level, low angle, overhead)",
    "depth_of_field": "shallow, deep, or uniform focus",
    "focus": "what is in sharp focus",
    "lens_focal_length": "descriptive focal length (e.g. standard, telephoto)"
  },
  "style_medium": "visual style or medium (e.g. live action, animation, drone footage)",
  "artistic_style": "genre or artistic approach (e.g. documentary, cinematic, realistic)",
  "context": "scene context or use case",
  "text_and_signage_elements": ["visible text, signs, or overlays in the scene"],
  "actions": [{"time": "M:SS-M:SS", "description": "timed event description"}],
  "segments": [
    {
      "segment_index": 0,
      "time_range": "M:SS-M:SS",
      "description": "what happens in this segment",
      "key_changes": "notable changes within segment",
      "camera": "camera behavior in segment"
    }
  ],
  "transitions": ["transition descriptions between segments, or empty if single take"]
}"""

_TEMPLATE_TRANSFER_V4_2 = (
    "Your output must be a single JSON object with exactly these top-level keys:\n"
    + _TEMPLATE_TRANSFER_OUTPUT_JSON_V4_2
    + "\n\n"
)

_TEMPLATE_TRANSFER_STRUCTURED_V4_2 = r"""<instructions>
You are a prompt upsampler for a transfer-conditioned image/video generation model. This instructions block defines the task and binds the output contract. The full input consists of this instructions block, attached transfer-conditioning media, a <video_description> scene description, a <task_constraints> numbered constraint list, and an <output_json_template> JSON schema.

The attached media are supplied by the caller before this text:
- For transfer_mode="t2v": attached control video(s), then this text prompt.
- For transfer_mode="i2v": attached starting frame image, attached control video(s), then this text prompt.
- For transfer_mode="v2v": attached source-video opening clip, attached control video(s), then this text prompt.

{transfer_instruction}

Produce exactly one JSON object. The object must fully populate every required field in the output template and strictly satisfy every numbered task constraint. The output JSON describes the target video in dense structured form, consistent with both the text description and attached transfer-conditioning media.
</instructions>

<video_description>
{description}
</video_description>

<task_constraints>
1. **Transfer conditioning.** Use the attached control video(s) as structural and motion guidance. The output JSON must be consistent with the visible layout, geometry, subject placement, motion boundaries, and temporal evolution implied by the control video(s).

2. **Mode grounding.**
   - For transfer_mode="t2v", rely on the text description plus control video(s).
   - For transfer_mode="i2v", keep the attached starting frame as the visual anchor for subjects, setting, composition, lighting, and style.
   - For transfer_mode="v2v", keep the attached source-video opening clip as the visual anchor for the beginning of the generated video, including subject identity, scene layout, camera style, and initial motion.

3. **Control consistency.** Do not invent major geometry, object placements, camera motion, or subject movements that conflict with the attached control video(s). If multiple controls are attached, satisfy all of them as much as possible:
   - edge constrains outlines and object boundaries.
   - blur constrains coarse appearance and low-frequency layout.
   - depth constrains 3D structure, distance, and foreground/background layout.
   - seg constrains object/region identity and spatial grouping.
   - world_scenario constrains AV scene layout and world-state semantics.

4. **Faithfulness.** Do not contradict the provided video description or attached media. Every concrete described element, including subjects, key actions, props, setting cues, camera cues, and visible text, must appear in the JSON either directly or as a clearly plausible extension that does not change the core meaning.

5. **Dense JSON, not a short caption.** Expand the prompt into a complete structured annotation. Add rich scene detail when helpful, but keep additions physically plausible and grounded in the description and conditioning media.

6. **Temporal consistency.** Make `actions`, `segments`, subject `action`, subject `state_changes`, and `cinematography.camera_motion` agree with each other. Use M:SS-M:SS time ranges for `actions[].time` and `segments[].time_range`.

7. **Schema completeness.** Include every top-level key from the template exactly once. Do not invent extra top-level keys. Populate every field with specific, scene-justified detail. Use empty values only when truly inapplicable.

8. **Allowed empty values.** The only allowed empty values are exactly: "", 0, [], and {}. Do not use null.

9. **Output format.** Return only a single JSON object. Do not include prose, explanations, markdown headings, or any text outside the JSON.
</task_constraints>

<output_json_template>
{output_json_template}
</output_json_template>"""

_TRANSFER_CONTROL_MODALITY_PHRASES: dict[str, str] = {
    "edge": "Canny-edge control video",
    "blur": "blurred control video",
    "depth": "per-pixel depth control video",
    "seg": "color-coded segmentation control video",
    "world_scenario": "AV world-scenario control video",
}

_TRANSFER_INSTRUCTION_FAMILIES_V4_2: dict[str, tuple[str, ...]] = {
    "t2v": (
        "Given the following text description of a video and the attached {ctrl}, produce a comprehensive, structured JSON capturing all visual elements, subjects, actions, cinematography, lighting, aesthetics, and temporal progression consistent with the control video.\n\n",
        "Convert the following video description into detailed, structured JSON format that captures every visual and temporal element. The attached {ctrl} provides the scene layout.\n\n",
        "Given the following video description and the attached {ctrl}, produce a complete, structured JSON documenting subjects, background, lighting, cinematography, actions, and temporal flow.\n\n",
        "Given the following description and the attached {ctrl}, output a fully detailed JSON capturing all subjects, their appearances, actions, the setting, camera work, and time-based progression.\n\n",
        "Using the attached {ctrl} as structural reference, convert the following video description into structured JSON data covering subjects, environment, cinematography, lighting, color aesthetics, and temporal events.\n\n",
        "Transform the following video description into rich, hierarchical JSON. The attached {ctrl} defines the spatial arrangement. Include complete subject details, scene context, technical cinematography parameters, and temporal analysis.\n\n",
        "Given the following video description and the attached {ctrl}, output a dense, structured JSON covering all subjects, the background and setting, lighting, camera movements, aesthetic style, and action sequences.\n\n",
        "Convert the following video description into a complete, structured JSON document. The attached {ctrl} provides structural context for the scene layout and motion.\n\n",
        "Given the following video description and the attached {ctrl}, respond with a single JSON object covering subjects, setting, lighting, cinematography, aesthetics, actions, segments, and temporal progression. Be thorough and precise.\n\n",
        "Convert the following video description into a dense, well-structured JSON. The attached {ctrl} shows the scene structure. Capture every subject, the scene background, lighting, camera behavior, color palette, mood, and temporal progression.\n\n",
    ),
    "i2v": (
        "Using the attached starting frame, the attached {ctrl}, and the following text description, produce a comprehensive, structured JSON capturing all visual elements consistent with both the frame and the control video.\n\n",
        "The attached image is the first frame of the video. Combined with the attached {ctrl} and the description below, generate detailed structured JSON covering subjects, background, lighting, cinematography, actions, and temporal flow.\n\n",
        "Given the attached starting frame for visual ground truth, the attached {ctrl} for scene structure, and the following description, produce a complete structured JSON document.\n\n",
        "Using the first frame (attached image) and the attached {ctrl} as references, convert the following video description into dense, structured JSON capturing every subject, lighting, camera behavior, and temporal progression.\n\n",
        "The image shows the starting frame. The attached {ctrl} provides structural context. Using these and the description below, output a fully detailed JSON.\n\n",
        "Given the attached starting frame, the attached {ctrl}, and the following description, transform the text into rich hierarchical JSON with complete subject details, scene context, and temporal analysis.\n\n",
        "Using the attached starting frame and {ctrl}, convert the following video description into structured JSON data. Produce detailed output covering subjects, environment, cinematography, lighting, and temporal events.\n\n",
        "The attached image is the opening frame and the attached {ctrl} defines the scene layout. Given the description below, respond with a single JSON object. Be thorough.\n\n",
        "Given the starting frame (attached image), the attached {ctrl}, and the video description below, output a dense structured JSON covering all subjects and their visual details, the setting, lighting, camera movements, and action sequences.\n\n",
        "Using the first frame and the attached {ctrl} as visual references, convert the following description into a complete structured JSON document capturing every element of the video.\n\n",
    ),
    "v2v": (
        "The attached short source-video clip is the opening of the requested video. Combined with the attached {ctrl} and the description below, produce a comprehensive, structured JSON.\n\n",
        "Using the attached source-video opening clip, the attached {ctrl}, and the following description, generate detailed structured JSON covering every visual and temporal element.\n\n",
        "The attached video clip shows how the video begins. The attached {ctrl} provides structural context. Given the description below, produce a complete structured JSON.\n\n",
        "Given the attached source-video clip, the attached {ctrl} for scene structure, and the following text description, output a fully detailed JSON capturing subjects, actions, the setting, camera work, and temporal progression.\n\n",
        "Using the attached opening clip and {ctrl}, convert the following video description into structured JSON data covering subjects, environment, cinematography, and temporal events.\n\n",
        "The attached source-video clip and {ctrl} provide visual and structural context. Transform the following description into rich hierarchical JSON with complete details.\n\n",
        "Given the attached source-video clip and {ctrl}, convert the following video description into a dense structured JSON covering all subjects, lighting, camera movements, and temporal progression.\n\n",
        "Using the opening video clip and the attached {ctrl} as references, respond with a single JSON object covering subjects, setting, lighting, cinematography, aesthetics, actions, and segments. Be thorough.\n\n",
        "The attached clip is the source video opening. The attached {ctrl} shows the scene structure. Convert the description below into a complete structured JSON document.\n\n",
        "Given the attached source video opening, the attached {ctrl}, and the following description, output a well-structured JSON capturing every subject, the scene background, lighting, camera behavior, and the scene's temporal progression.\n\n",
    ),
}


# Registry of every (version, task) → template body. `build_user_text` /
# `build_messages` select an entry via the `version=` kwarg (default
# "v4.2"). For t2i, the v4.2 default body is the EXPRESSIVE variant — the
# v4.2-baseline body remains addressable as ("v4.2-original", "t2i") for
# baseline / A-B comparisons. The constrained anti-hallucination variant is
# at ("v4.2-constrained", "t2i"). Transfer has an analogous structured
# variant at ("v4.2-structured", "transfer").
CANONICAL_TEMPLATES: dict[tuple[str, str], str] = {
    ("v4.2", "t2v"): _TEMPLATE_T2V_V4_2,
    ("v4.2", "t2i"): _TEMPLATE_T2I_V4_2_EXPRESSIVE,
    ("v4.2", "i2v"): _TEMPLATE_I2V_V4_2,
    ("v4.2", "transfer"): _TEMPLATE_TRANSFER_V4_2,
    ("v4.2-structured", "transfer"): _TEMPLATE_TRANSFER_STRUCTURED_V4_2,
    ("v4.2-expressive", "t2i"): _TEMPLATE_T2I_V4_2_EXPRESSIVE,
    ("v4.2-original", "t2i"): _TEMPLATE_T2I_V4_2,
    ("v4.2-constrained", "t2i"): _TEMPLATE_T2I_V4_2_CONSTRAINED,
}


def _format_transfer_control_modalities(control_modalities: str | list[str]) -> str:
    """Build the transfer-control phrase used by the existing transfer prompt."""
    if isinstance(control_modalities, str):
        stripped = control_modalities.strip()
        if not stripped:
            raise ValueError("`control_modalities` must be a non-empty string or list")
        return _TRANSFER_CONTROL_MODALITY_PHRASES.get(stripped, stripped)

    if not control_modalities:
        raise ValueError("`control_modalities` must contain at least one modality")

    phrases = []
    for modality in control_modalities:
        if modality not in _TRANSFER_CONTROL_MODALITY_PHRASES:
            valid = ", ".join(sorted(_TRANSFER_CONTROL_MODALITY_PHRASES))
            raise ValueError(f"unknown transfer control modality {modality!r}; expected one of: {valid}")
        phrases.append(_TRANSFER_CONTROL_MODALITY_PHRASES[modality])

    if len(phrases) == 1:
        return phrases[0]
    return " and ".join([", ".join(phrases[:-1]), phrases[-1]])


def _build_transfer_user_text(
    description: str,
    *,
    transfer_mode: str,
    control_modalities: str | list[str],
    version: str,
    transfer_instruction_index: int,
) -> str:
    """Render the existing transfer prompt: T2V schema skeleton + transfer instruction + NL input."""
    if transfer_mode not in _TRANSFER_INSTRUCTION_FAMILIES_V4_2:
        valid = ", ".join(sorted(_TRANSFER_INSTRUCTION_FAMILIES_V4_2))
        raise ValueError(f"unknown transfer_mode {transfer_mode!r}; expected one of: {valid}")

    instruction_templates = _TRANSFER_INSTRUCTION_FAMILIES_V4_2[transfer_mode]
    if not 0 <= transfer_instruction_index < len(instruction_templates):
        raise ValueError(
            f"transfer_instruction_index must be in [0, {len(instruction_templates) - 1}], "
            f"got {transfer_instruction_index}"
        )

    ctrl_phrase = _format_transfer_control_modalities(control_modalities)
    instruction = instruction_templates[transfer_instruction_index].format(ctrl=ctrl_phrase).strip()
    template = CANONICAL_TEMPLATES[(version, "transfer")]
    if "{transfer_instruction}" in template:
        return (
            template.replace("{transfer_instruction}", instruction)
            .replace("{description}", description)
            .replace("{output_json_template}", _TEMPLATE_TRANSFER_OUTPUT_JSON_V4_2.strip())
        )
    return template + instruction + "\n\n" + description


def _format_duration(duration_secs: int) -> str:
    """Render duration seconds as M:SS (e.g. 6 -> '0:06', 75 -> '1:15')."""
    return f"{duration_secs // 60}:{duration_secs % 60:02d}"


def build_user_text(
    task: str,
    description: str,
    *,
    aspect_ratio: str | None = None,
    resolution_w: int | None = None,
    resolution_h: int | None = None,
    fps: int | None = None,
    duration_secs: int | None = None,
    transfer_mode: str | None = None,
    control_modalities: str | list[str] | None = None,
    transfer_instruction_index: int = 0,
    version: str = "v4.2",
) -> str:
    """Returns the canonical user-text prompt for the upsampler.

    Args:
        task: ``"t2v"`` (text-to-video), ``"t2i"`` (text-to-image), or
            ``"i2v"`` (image-to-video), or ``"transfer"``.
        description: source scene description to be upsampled. Inserted into
            the ``<video_description>`` (or ``<image_description>`` for t2i)
            block of the canonical template. For transfer, this is appended
            after the rendered transfer instruction, matching the existing
            transfer SFT prompt format.
        aspect_ratio: clip aspect ratio in comma form, e.g. ``"1,1"``,
            ``"16,9"``, ``"9,16"``, ``"4,3"``, ``"3,4"``. Injected into the
            ``aspect_ratio`` constraint of the template (required for
            non-transfer tasks).
        resolution_w: output frame width in pixels (required for non-transfer tasks).
        resolution_h: output frame height in pixels (required for non-transfer tasks).
        fps: target frames-per-second (required for ``t2v`` and ``i2v``; must
            be ``None`` or omitted for ``t2i``).
        duration_secs: clip duration in whole seconds (required for ``t2v``
            and ``i2v``; must be ``None`` or omitted for ``t2i``). Rendered as
            ``M:SS`` in the template.
        transfer_mode: transfer generation mode, one of ``"t2v"``, ``"i2v"``,
            or ``"v2v"``. Required when ``task="transfer"``.
        control_modalities: transfer control modality list, e.g.
            ``["depth", "seg"]``. A pre-rendered control phrase is also
            accepted as a string. Required when ``task="transfer"``.
        transfer_instruction_index: which existing transfer instruction variant
            to render for the selected mode. Defaults to 0 for deterministic
            inference-team use.
        version: canonical version label. Default ``"v4.2"``. For transfer,
            ``"v4.2"`` preserves the existing SFT prompt shape and
            ``"v4.2-structured"`` wraps the same transfer contract in the
            shared upsampler tag structure.

    Returns:
        Full user-text string with ``<instructions>``, the description block,
        ``<task_constraints>``, and ``<output_json_template>`` sections, with
        the (description, fps, duration, aspect_ratio, resolution) values
        substituted dynamically. For transfer, returns either the existing SFT
        prompt shape or the structured-tagged variant, depending on ``version``.

    Raises:
        KeyError: if no canonical exists for ``(version, task)``.
        ValueError: if required task-specific parameters are missing or invalid.
    """
    if task == "transfer":
        if transfer_mode is None:
            raise ValueError("task='transfer' requires `transfer_mode`")
        if control_modalities is None:
            raise ValueError("task='transfer' requires `control_modalities`")
        return _build_transfer_user_text(
            description,
            transfer_mode=transfer_mode,
            control_modalities=control_modalities,
            version=version,
            transfer_instruction_index=transfer_instruction_index,
        )

    if aspect_ratio is None or resolution_w is None or resolution_h is None:
        raise ValueError(
            f"task={task!r} requires `aspect_ratio`, `resolution_w`, and `resolution_h`; "
            f"got aspect_ratio={aspect_ratio!r}, resolution_w={resolution_w!r}, resolution_h={resolution_h!r}"
        )

    is_video = task in ("t2v", "i2v")
    if is_video and (fps is None or duration_secs is None):
        raise ValueError(
            f"task={task!r} requires both `fps` and `duration_secs`; got fps={fps!r}, duration_secs={duration_secs!r}"
        )
    template = CANONICAL_TEMPLATES[(version, task)]
    text = template.replace("{description}", description)
    text = text.replace("{aspect_ratio}", aspect_ratio)
    text = text.replace("{resolution_w}", str(resolution_w))
    text = text.replace("{resolution_h}", str(resolution_h))
    if is_video:
        if duration_secs is None:
            raise ValueError(f"task={task!r} requires `duration_secs`; got duration_secs={duration_secs!r}")
        if fps is None:
            raise ValueError(f"task={task!r} requires `fps`; got fps={fps!r}")
        text = text.replace("{duration}", _format_duration(duration_secs))
        text = text.replace("{fps}", str(fps))
    else:
        if duration_secs is not None:
            raise ValueError(f"task={task!r} does not require `duration_secs`; got duration_secs={duration_secs!r}")
        if fps is not None:
            raise ValueError(f"task={task!r} does not require `fps`; got fps={fps!r}")
    return text


def build_messages(
    task: str,
    description: str,
    *,
    aspect_ratio: str | None = None,
    resolution_w: int | None = None,
    resolution_h: int | None = None,
    fps: int | None = None,
    duration_secs: int | None = None,
    transfer_mode: str | None = None,
    control_modalities: str | list[str] | None = None,
    transfer_instruction_index: int = 0,
    version: str = "v4.2",
) -> list[dict]:
    """Returns OpenAI chat-completions-style messages list.

    For text-only tasks (``t2v``, ``t2i``) this returns ``[system, user]``
    with a string user-content. For image-to-video (``i2v``) the caller is
    responsible for adding the image to the user content alongside the
    returned text. For transfer, the caller is responsible for supplying the
    mode-specific media and control video(s) alongside the returned text. This
    function returns the text-only canonical and does not embed media bytes.
    """
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": build_user_text(
                task,
                description,
                aspect_ratio=aspect_ratio,
                resolution_w=resolution_w,
                resolution_h=resolution_h,
                fps=fps,
                duration_secs=duration_secs,
                transfer_mode=transfer_mode,
                control_modalities=control_modalities,
                transfer_instruction_index=transfer_instruction_index,
                version=version,
            ),
        },
    ]


# ----- Response post-processing ------------------------------------------- #
# Defensive cleaner for thinking-style outputs. Canonical V4.2 SFT models
# (cosmos3 upsampler 8B / 32B) emit clean fenced JSON with no reasoning
# preamble and no <think> markers; the cleaner is a no-op on those outputs.
# For reasoning-style backbones (Qwen3-reasoning, DeepSeek-R1, etc.) that
# emit ``<think>...</think>`` blocks before the JSON fence, ``clean_response``
# strips the markers + any reasoning preamble so the downstream consumer
# sees only the fenced JSON object.

_THINKING_TAG_PATTERNS = [
    _re.compile(r"<think>.*?</think>", _re.S | _re.I),
    _re.compile(r"<thought>.*?</thought>", _re.S | _re.I),
    _re.compile(r"<thinking>.*?</thinking>", _re.S | _re.I),
    _re.compile(r"<reasoning>.*?</reasoning>", _re.S | _re.I),
    _re.compile(r"<scratch>.*?</scratch>", _re.S | _re.I),
    _re.compile(r"<reflection>.*?</reflection>", _re.S | _re.I),
]


def clean_response(content: str, strip_think_when_appears: bool = True) -> "tuple[str, dict]":
    """Strip thinking-style markers from an upsampler response.

    Trigger rule (LOCKED 2026-05-17): post-processing fires ONLY when at
    least one of ``<think>``/``<thought>``/``<thinking>``/``<reasoning>``/
    ``<scratch>``/``<reflection>`` is detected in ``content``. If no such
    tag is present, the content is returned BYTE-FOR-BYTE unchanged — no
    whitespace strip, no preamble trim, no JSON re-fencing. This guarantees
    the cleaner cannot bias eval outputs from canonical SFT models (which
    already emit clean fenced JSON).

    When a tag IS detected, the cleaner removes:
      1. every tagged thinking block (multi-line, case-insensitive);
      2. any non-empty preamble between the (now removed) tag header and
         the first ```json``` fence in the remaining content.

    Args:
        content: raw model output (possibly empty).
        strip_think_when_appears: master switch. ``True`` (default) enables
            the trigger rule above. ``False`` returns ``content`` unchanged.

    Returns:
        ``(cleaned_content, info_dict)`` where ``info_dict`` has:
          - ``tags_stripped`` (int): count of tagged blocks removed
          - ``tag_kinds`` (list[str]): tag types that were found
          - ``preamble_chars_stripped`` (int): chars removed before ```json```
          - ``was_clean`` (bool): True iff no stripping happened (no-op)

    Idempotent. Never raises.
    """
    info = {"tags_stripped": 0, "tag_kinds": [], "preamble_chars_stripped": 0, "was_clean": True}
    if not content or not strip_think_when_appears:
        return content, info
    has_any_tag = any(pat.search(content) for pat in _THINKING_TAG_PATTERNS)
    if not has_any_tag:
        return content, info
    clean = content
    for pat in _THINKING_TAG_PATTERNS:
        matches = pat.findall(clean)
        if matches:
            info["tags_stripped"] += len(matches)
            tag_name = pat.pattern.split(">")[0].lstrip("<")
            info["tag_kinds"].append(tag_name)
            info["was_clean"] = False
            clean = pat.sub("", clean)
    m = _re.search(r"```json", clean, _re.S)
    if m and m.start() > 0:
        preamble = clean[: m.start()].strip()
        if preamble:
            info["preamble_chars_stripped"] = len(preamble)
            info["was_clean"] = False
            clean = clean[m.start() :]
    return clean.strip(), info


def is_upsampled_prompt(prompt: str) -> bool:
    """Heuristic: return ``True`` iff ``prompt`` looks like the V4.2
    upsampler's emitted output and therefore should NOT be fed through
    the native upsampler again.

    Used by inference callers (e.g.
    ``cosmos3.inference.OmniInference._iter_predictions``) to decide
    per-batch whether to pass a native prompt-upsample task to
    :meth:`OmniMoTModel.generate_samples_from_batch`.  Two motivating
    cases produce already-upsampled prompts:

    1. The user (or an upstream pipeline) supplies a pre-upsampled
       caption directly as ``sample_args.prompt`` — typically the
       fenced ``\\`\\`\\`json {...} \\`\\`\\``` output of an earlier
       upsampler run that they want to reuse verbatim.
    2. The external endpoint path
       (``OmniSampleOverrides._apply_prompt_upsampling_local`` in
       ``packages/cosmos3/cosmos3/args.py``) has already replaced
       ``self.prompt`` with the endpoint's JSON output before
       ``generate_samples_from_batch`` runs; the native upsampler must
       not double-process that.

    Recognised shapes (after leading/trailing whitespace strip):

      - Fenced JSON:  ``\\`\\`\\`json ... \\`\\`\\```  (also accepts the
        ``\\`\\`\\`\\n{`` shorthand some backends emit when the
        language tag is omitted).
      - Bare JSON object: first non-whitespace char is ``{`` AND the
        whole string parses as a JSON dict via ``json.loads``.

    Everything else returns ``False`` and falls through to native
    upsampling: plain prose, empty / whitespace-only strings, JSON
    arrays / scalars (numbers, strings, booleans), malformed JSON, and
    fenced non-JSON blocks (``\\`\\`\\`python``, etc.).  The function
    is intentionally conservative — false negatives are safe (we just
    redundantly upsample), while false positives would silently skip
    needed upsampling.  Never raises.

    Args:
        prompt: candidate caption string.  ``None`` is tolerated for
            caller convenience and treated as ``""``.

    Returns:
        ``True`` iff the prompt looks like an already-upsampled
        V4.2 payload that should be passed through unchanged.
    """
    s = (prompt or "").strip()
    if not s:
        return False
    if s.startswith("```json") or s.startswith("```\n{"):
        return True
    if s.startswith("{"):
        try:
            obj = _json.loads(s)
        except (_json.JSONDecodeError, ValueError):
            return False
        return isinstance(obj, dict)
    return False


# ----- Self-test ---------------------------------------------------------- #
# Run as ``python -m cosmos_framework.model.generator.upsampler.prompts`` to verify that
# the templates round-trip and the ``{description}`` placeholder is honoured.


def _self_test() -> None:
    sample_desc = "A cat playing with yarn under afternoon sunlight."
    # Per-task realistic (W, H, aspect, fps, duration) — different per task
    # to verify each template substitutes its own values, not a hardcoded one.
    PARAMS = {
        "t2v": dict(aspect_ratio="16,9", resolution_w=1280, resolution_h=720, fps=24, duration_secs=5),
        "t2i": dict(aspect_ratio="4,3", resolution_w=1280, resolution_h=960),
        "i2v": dict(aspect_ratio="9,16", resolution_w=480, resolution_h=832, fps=30, duration_secs=10),
    }
    for task, params in PARAMS.items():
        text = build_user_text(task=task, description=sample_desc, **params)
        assert sample_desc in text, f"description not injected for {task}"
        assert "{description}" not in text, f"placeholder leaked for {task}"
        # Each canonical has the four required XML tag blocks
        assert "<instructions>" in text and "</instructions>" in text
        assert "<task_constraints>" in text and "</task_constraints>" in text
        assert "<output_json_template>" in text and "</output_json_template>" in text
        if task == "t2i":
            assert "<image_description>" in text
        else:
            assert "<video_description>" in text
        # Verify NO placeholders leaked
        for placeholder in ("{aspect_ratio}", "{resolution_w}", "{resolution_h}"):
            assert placeholder not in text, f"{placeholder} leaked for {task}"
        if task != "t2i":
            assert "{duration}" not in text, f"{{duration}} leaked for {task}"
            assert "{fps}" not in text, f"{{fps}} leaked for {task}"
        # Verify dynamic values ARE present
        assert f'aspect_ratio: "{params["aspect_ratio"]}"' in text, f"aspect_ratio not substituted for {task}"
        assert (f'resolution: {{"W": {params["resolution_w"]}, "H": {params["resolution_h"]}}}') in text, (
            f"resolution not substituted for {task}"
        )
        if task != "t2i":
            expected_dur = _format_duration(params["duration_secs"])
            assert f'duration: "{expected_dur}"' in text, f"duration not substituted for {task}"
            assert f"fps: {params['fps']}" in text, f"fps not substituted for {task}"
        # Roundtrip via build_messages
        msgs = build_messages(task=task, description=sample_desc, **params)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == text
        print(
            f"[OK] {task}: user_text={len(text)} chars, "
            f"aspect={params['aspect_ratio']}, "
            f"res={params['resolution_w']}x{params['resolution_h']}"
            + (f", fps={params['fps']}, dur={_format_duration(params['duration_secs'])}" if task != "t2i" else "")
        )
    transfer_text = build_user_text(
        task="transfer",
        description=sample_desc,
        transfer_mode="v2v",
        control_modalities=["depth", "seg"],
        transfer_instruction_index=3,
    )
    assert sample_desc in transfer_text
    assert "{description}" not in transfer_text
    assert "Your output must be a single JSON object with exactly these top-level keys" in transfer_text
    assert (
        "Given the attached source-video clip, the attached per-pixel depth control video and "
        "color-coded segmentation control video for scene structure"
    ) in transfer_text
    transfer_structured_text = build_user_text(
        task="transfer",
        description=sample_desc,
        transfer_mode="v2v",
        control_modalities=["depth", "seg"],
        transfer_instruction_index=3,
        version="v4.2-structured",
    )
    assert sample_desc in transfer_structured_text
    assert "{description}" not in transfer_structured_text
    assert "{transfer_instruction}" not in transfer_structured_text
    assert "{output_json_template}" not in transfer_structured_text
    transfer_section_tags = (
        "<instructions>\n",
        "\n</instructions>\n",
        "\n<video_description>\n",
        "\n</video_description>\n",
        "\n<task_constraints>\n",
        "\n</task_constraints>\n",
        "\n<output_json_template>\n",
        "\n</output_json_template>",
    )
    transfer_tag_positions = [transfer_structured_text.find(tag) for tag in transfer_section_tags]
    assert all(position >= 0 for position in transfer_tag_positions)
    assert transfer_tag_positions == sorted(transfer_tag_positions)
    assert "Transfer conditioning." in transfer_structured_text
    assert "Mode grounding." in transfer_structured_text
    output_template = transfer_structured_text.split("<output_json_template>\n", 1)[1].split(
        "\n</output_json_template>", 1
    )[0]
    assert output_template == _TEMPLATE_TRANSFER_OUTPUT_JSON_V4_2.strip()
    assert "Your output must be a single JSON object with exactly these top-level keys" not in output_template
    transfer_msgs = build_messages(
        task="transfer",
        description=sample_desc,
        transfer_mode="v2v",
        control_modalities=["depth", "seg"],
        transfer_instruction_index=3,
    )
    assert len(transfer_msgs) == 2
    assert transfer_msgs[1]["content"] == transfer_text
    print("[OK] transfer: legacy skeleton prompt + structured tagged variant")
    # Negative test: video task without fps/duration should raise
    try:
        build_user_text(task="t2v", description=sample_desc, aspect_ratio="1,1", resolution_w=640, resolution_h=640)
    except ValueError:
        print("[OK] t2v without fps/duration_secs raises ValueError")
    else:
        raise AssertionError("expected ValueError for t2v without fps/duration_secs")
    # Duration formatting
    assert _format_duration(6) == "0:06"
    assert _format_duration(75) == "1:15"
    assert _format_duration(0) == "0:00"
    print("[OK] _format_duration: 6 -> 0:06, 75 -> 1:15, 0 -> 0:00")

    # clean_response — canonical (clean) outputs must be byte-for-byte unchanged
    canonical_clean = '```json\n{"scene_imagination": "x", "duration": "0:05"}\n```'
    out, info = clean_response(canonical_clean)
    assert out == canonical_clean, "clean_response modified clean output"
    assert info["was_clean"] is True
    assert info["tags_stripped"] == 0
    # Adversarial: word "think" inside JSON without tags = still byte-unchanged
    adversarial = '```json\n{"desc": "the word think appears here"}\n```'
    out2, info2 = clean_response(adversarial)
    assert out2 == adversarial, "clean_response false-triggered on bare word"
    assert info2["was_clean"] is True
    # Dirty: <think>...</think> + clean fenced JSON = tag stripped, fence preserved
    dirty = "<think>\nLet me reason...\n</think>\n\n" + canonical_clean
    out3, info3 = clean_response(dirty)
    assert info3["tags_stripped"] == 1
    assert info3["was_clean"] is False
    assert "<think>" not in out3 and "</think>" not in out3
    assert out3.startswith("```json")
    # strip_think_when_appears=False -> always no-op
    out4, info4 = clean_response(dirty, strip_think_when_appears=False)
    assert out4 == dirty
    assert info4["was_clean"] is True
    # Idempotent
    assert clean_response(clean_response(dirty)[0])[0] == clean_response(dirty)[0]
    print("[OK] clean_response: clean+adversarial unchanged, dirty stripped, flag off=no-op, idempotent")

    # is_upsampled_prompt — positive cases (already-upsampled, must skip native pass).
    # Fenced JSON in both flavours that the V4.2 family emits.
    assert is_upsampled_prompt(canonical_clean), "fenced ```json``` must be detected"
    assert is_upsampled_prompt("```json\n{}\n```"), "fenced empty-dict JSON must be detected"
    assert is_upsampled_prompt('```\n{"k": 1}\n```'), "fenced JSON without language tag must be detected"
    # Bare JSON object — common when a backend strips the fence.
    assert is_upsampled_prompt('{"temporal_caption": "...", "duration": "0:05"}'), "bare JSON dict must be detected"
    assert is_upsampled_prompt("  {}  "), "bare empty-dict JSON with surrounding whitespace must be detected"
    # is_upsampled_prompt — negative cases (plain content, must run native upsampling).
    assert not is_upsampled_prompt(""), "empty string must NOT be detected"
    assert not is_upsampled_prompt("   "), "whitespace-only must NOT be detected"
    assert not is_upsampled_prompt("A black cat sits on a windowsill."), "plain prose must NOT be detected"
    assert not is_upsampled_prompt("[1, 2, 3]"), "JSON array (non-dict) must NOT be detected"
    assert not is_upsampled_prompt('"a string"'), "JSON scalar string must NOT be detected"
    assert not is_upsampled_prompt("42"), "JSON scalar number must NOT be detected"
    assert not is_upsampled_prompt("{not valid json"), "malformed JSON must NOT be detected"
    assert not is_upsampled_prompt("```python\nprint(1)\n```"), "fenced non-JSON block must NOT be detected"
    assert not is_upsampled_prompt(None), "None must be tolerated and NOT be detected"  # type: ignore[arg-type]
    print("[OK] is_upsampled_prompt: fenced+bare JSON detected; prose/arrays/scalars/malformed/non-JSON fences ignored")

    print("All canonical templates self-test passed.")


if __name__ == "__main__":
    _self_test()
