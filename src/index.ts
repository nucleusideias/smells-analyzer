import fs from "fs";
import pdfParse from "pdf-parse";
import { google } from "@ai-sdk/google";
import { generateObject } from "ai";
import { z } from "zod";

import { embedAndStore } from "./generateEmbeddings";

import "dotenv/config";

const MODEL_NAME = "gemini-1.5-pro";

async function extractPdfText(pdfPath: string): Promise<string> {
  const dataBuffer = fs.readFileSync(pdfPath);
  const data = await pdfParse(dataBuffer);
  return data.text;
}

async function extractCodeSmells(text: string) {
  const prompt = `
    Extract code smells from the following text. For each, provide:
    - category
    - name
    - description
    - a Java code example

    Text:
    ${text}
  `;

  const result = await generateObject({
    model: google(MODEL_NAME),
    schema: z.array(
      z.object({
        category: z.string().describe("The category of the code smell"),
        name: z.string().describe("The name of the code smell"),
        description: z
          .string()
          .describe(
            "The description of the code smell, if possible, add causes and symptoms of the code smell"
          ),
        java_example: z
          .string()
          .describe("A Java code example of the code smell"),
      })
    ),
    prompt,
  });

  return result.object;
}

async function insertCodeSmells(
  smells: {
    category: string;
    name: string;
    description: string;
    java_example: string;
  }[]
): Promise<void> {
  try {
    for (const smell of smells) {
      const result = await embedAndStore(
        {
          description: smell.description,
          name: smell.name,
          category: smell.category,
          java_example: smell.java_example,
        },
        false,
        "code_smells"
      );

      if (!result.success) {
        console.error("Error embedding code smell:", result.error);
      }
    }
  } catch (error) {
    console.error("Error inserting code smells:", error);
  }
}

async function saveText(text: string, fileName: string) {
  fs.writeFileSync(`src/database/outputs/${fileName}.txt`, text);
}

async function main() {
  const files = [
    "src/database/Bloaters.pdf",
    "src/database/Object-Orientation Abusers.pdf",
    "src/database/Change Preventers.pdf",
    "src/database/Dispensables.pdf",
    "src/database/Acopladores.pdf",
  ];

  const smells = await Promise.all(
    files.map(async (file, i) => {
      const text = await extractPdfText(file);
      const smells = await extractCodeSmells(text);
      await saveText(
        JSON.stringify(smells),
        file.split("/").pop()?.split(".")?.at(0) || `output-${i}`
      );

      return smells;
    })
  );

  for (const smell of smells) {
    await insertCodeSmells(smell);
  }
}

main();
