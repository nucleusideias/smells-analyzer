import "dotenv/config";
import { createClient } from "@supabase/supabase-js";

import { google } from "@ai-sdk/google";
import { embed } from "ai";

const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_KEY!
);

const EMBEDDING_MODEL = google.textEmbeddingModel("text-embedding-004");

const RAW_VECTOR_DIMENSIONS = 768;
const OPTIMIZED_VECTOR_DIMENSIONS = 256;

function normalizeL2(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
  if (norm === 0) return v;
  return v.map((val) => val / norm);
}

function optimizeEmbedding(
  embedding: number[],
  dimension: number = embedding.length
): number[] {
  if (dimension < 0 || dimension > embedding.length) {
    console.warn(
      `Invalid target dimension ${dimension}. Using original length ${embedding.length}.`
    );
    dimension = embedding.length;
  }

  const truncated =
    dimension === embedding.length ? embedding : embedding.slice(0, dimension);

  const normalized = normalizeL2(truncated);

  if (normalized.length !== dimension) {
    console.warn(
      `Optimization resulted in unexpected dimension: ${normalized.length}`
    );
  }
  return normalized;
}

export async function embedAndStore(
  data: {
    description: string;
    name: string;
    category: string;
    java_example: string;
  },
  optimize: boolean = false,
  tableName: string = "code_smells"
): Promise<{ success: boolean; error?: Error }> {
  const cleanedText = data.description;

  try {
    console.log(
      `Generating Gemini embedding for: "${cleanedText.substring(0, 60)}..."`
    );
    const { embedding: rawEmbedding } = await embed({
      model: EMBEDDING_MODEL,
      value: cleanedText,
    });

    if (rawEmbedding.length !== RAW_VECTOR_DIMENSIONS && !optimize) {
      console.warn(
        `Expected ${RAW_VECTOR_DIMENSIONS} dimensions, got ${rawEmbedding.length}`
      );
    }

    const finalEmbedding = optimize
      ? optimizeEmbedding(rawEmbedding, OPTIMIZED_VECTOR_DIMENSIONS)
      : optimizeEmbedding(rawEmbedding);

    const targetDimension = optimize
      ? OPTIMIZED_VECTOR_DIMENSIONS
      : RAW_VECTOR_DIMENSIONS;
    if (finalEmbedding.length !== targetDimension) {
      throw new Error(
        `Final embedding dimension mismatch: expected ${targetDimension}, got ${finalEmbedding.length}`
      );
    }

    console.log(`Storing embedding with ${finalEmbedding.length} dimensions.`);
    const { error } = await supabase.from(tableName).insert({
      description: cleanedText,
      embedding: finalEmbedding,
      category: data.category,
      java_example: data.java_example,
      name: data.name,
    });

    if (error) {
      console.error("Supabase insert error:", error);
      throw new Error(
        `Failed to store embedding in Supabase: ${error.message}`
      );
    }

    console.log("Successfully stored text and embedding.");
    return { success: true };
  } catch (error: any) {
    console.error("Error in embedAndStore process:", error);
    return { success: false, error: error as Error };
  }
}
