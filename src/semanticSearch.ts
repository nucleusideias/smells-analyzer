import { google } from "@ai-sdk/google";
import { embed } from "ai";
import { createClient } from "@supabase/supabase-js";

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
    dimension = embedding.length;
  }
  const truncated =
    dimension === embedding.length ? embedding : embedding.slice(0, dimension);
  const normalized = normalizeL2(truncated);
  if (normalized.length !== dimension) {
    console.warn(
      `Search query optimization resulted in unexpected dimension: ${normalized.length}`
    );
  }
  return normalized;
}

export async function semanticSearch(
  query: string,
  optimize: boolean = false,
  limit: number = 5,
  threshold: number = 0.7
): Promise<{
  results: Array<{ content: string; similarity: number }>;
  error?: Error;
}> {
  try {
    const { embedding: rawQueryEmbedding } = await embed({
      model: EMBEDDING_MODEL,
      value: query,
    });

    const queryEmbedding = optimize
      ? optimizeEmbedding(rawQueryEmbedding, OPTIMIZED_VECTOR_DIMENSIONS)
      : optimizeEmbedding(rawQueryEmbedding); // Normalize only

    const expectedDimension = optimize
      ? OPTIMIZED_VECTOR_DIMENSIONS
      : RAW_VECTOR_DIMENSIONS;
    if (queryEmbedding.length !== expectedDimension) {
      throw new Error(
        `Query embedding dimension mismatch: expected ${expectedDimension}, got ${queryEmbedding.length}`
      );
    }

    const { data, error } = await supabase.rpc("match_documents", {
      query_embedding: queryEmbedding,
      match_threshold: threshold,
      match_count: limit,
    });

    if (error) {
      console.error("Supabase RPC error:", error);
      throw new Error(`Semantic search failed: ${error.message}`);
    }

    return { results: data || [] };
  } catch (error: any) {
    console.error("Error in semanticSearch process:", error);
    return { results: [], error: error as Error };
  }
}
