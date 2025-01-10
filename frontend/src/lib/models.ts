// Keep in sync with backend (llm.py)
// Order here matches dropdown order
export enum CodeGenerationModel {
  GEMINI_2_0_FLASH_THINKING= "gemini-2.0-flash-thinking-exp-1219",
  GEMINI_2_0_FLASH= "gemini-2.0-flash-exp",
  GEMINI_1_5_FLASH_8_b = "gemini-1.5-flash-8b",
  GEMINI_1_5_FLASH = "gemini-1.5-flash",
  GEMINI_1_5_PRO = "gemini-1.5-pro",
  GPT_4O = "gpt-4o",
  GPT_4_TURBO = "gpt-4-turbo",
  GPT_4_VISION = "gpt_4_vision",
  CLAUDE_3_SONNET = "claude_3_sonnet",

}

// Will generate a static error if a model in the enum above is not in the descriptions
export const CODE_GENERATION_MODEL_DESCRIPTIONS: {
  [key in CodeGenerationModel]: { name: string; inBeta: boolean };
} = {
  "gemini-2.0-flash-thinking-exp-1219": { name: "Gemini 2.0 Flash thinking", inBeta: false },
  "gemini-2.0-flash-exp": { name: "Gemini 2.0 Flash", inBeta: false },
  "gemini-1.5-flash-8b": { name: "Gemini 1.5 Flash 8b", inBeta: false },
  "gemini-1.5-flash": { name: "Gemini 1.5 Flash", inBeta: false },
  "gemini-1.5-pro": { name: "Gemini 1.5 PRO", inBeta: false },
  "gpt-4o": { name: "GPT-4o", inBeta: false },
  "gpt-4-turbo": { name: "GPT-4 Turbo (Apr 2024)", inBeta: false },
  gpt_4_vision: { name: "GPT-4 Vision (Nov 2023)", inBeta: false },
  claude_3_sonnet: { name: "Claude 3 Sonnet", inBeta: false },

};
