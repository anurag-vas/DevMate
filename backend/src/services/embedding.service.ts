import { GoogleGenerativeAI } from '@google/generative-ai'
import prisma from '../config/db'
import { env } from '../config/env'

const genAI = new GoogleGenerativeAI(env.GEMINI_API_KEY)
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' })

type ProfileForEmbedding = {
  id: string
  name: string
  bio?: string | null
  skills: string[]
  experience?: number | null
}

type JobForEmbedding = {
  id: string
  title: string
  description: string
  skills: string[]
  location?: string | null
  salary?: string | null
}

type VectorRow = {
  id: string
}

const toVectorLiteral = (embedding: number[]) => {
  return `[${embedding.join(',')}]`
}

export const createProfileEmbeddingText = (profile: {
  name: string
  bio?: string | null
  skills: string[]
  experience?: number | null
}) => {
  return [
    `Developer name: ${profile.name}`,
    `Skills: ${profile.skills.join(', ')}`,
    `Experience: ${profile.experience ?? 0} years`,
    `Bio: ${profile.bio || 'Not provided'}`
  ].join('\n')
}

export const createJobEmbeddingText = (job: {
  title: string
  description: string
  skills: string[]
  location?: string | null
  salary?: string | null
}) => {
  return [
    `Job title: ${job.title}`,
    `Description: ${job.description}`,
    `Required skills: ${job.skills.join(', ')}`,
    `Location: ${job.location || 'Not specified'}`,
    `Salary: ${job.salary || 'Not specified'}`
  ].join('\n')
}

export const generateEmbedding = async (text: string) => {
  const result = await embeddingModel.embedContent(text)
  return result.embedding.values
}

export const storeProfileEmbedding = async (profileId: string, embedding: number[]) => {
  await prisma.$executeRawUnsafe(
    'UPDATE "Profile" SET "embedding" = $1::vector WHERE "id" = $2',
    toVectorLiteral(embedding),
    profileId
  )
}

export const storeJobEmbedding = async (jobId: string, embedding: number[]) => {
  await prisma.$executeRawUnsafe(
    'UPDATE "Job" SET "embedding" = $1::vector WHERE "id" = $2',
    toVectorLiteral(embedding),
    jobId
  )
}

export const ensureProfileEmbedding = async (profile: ProfileForEmbedding) => {
  const embedding = await generateEmbedding(createProfileEmbeddingText(profile))
  await storeProfileEmbedding(profile.id, embedding)
  return embedding
}

export const ensureJobEmbedding = async (job: JobForEmbedding) => {
  const embedding = await generateEmbedding(createJobEmbeddingText(job))
  await storeJobEmbedding(job.id, embedding)
  return embedding
}

const backfillMissingProfileEmbeddings = async (limit: number) => {
  const rows: VectorRow[] = await prisma.$queryRawUnsafe(
    `
      SELECT "id"
      FROM "Profile"
      WHERE "embedding" IS NULL
      ORDER BY "createdAt" DESC
      LIMIT $1
    `,
    limit
  )

  if (rows.length === 0) {
    return
  }

  const profiles = await prisma.profile.findMany({
    where: {
      id: {
        in: rows.map((row: VectorRow) => row.id)
      }
    }
  })

  await Promise.all(
    profiles.map((profile: ProfileForEmbedding) => ensureProfileEmbedding(profile))
  )
}

export const findCandidateProfiles = async (
  jobEmbedding: number[],
  limit: number = 10
) => {
  await backfillMissingProfileEmbeddings(limit)

  const rows: VectorRow[] = await prisma.$queryRawUnsafe(
    `
      SELECT "id"
      FROM "Profile"
      WHERE "embedding" IS NOT NULL
      ORDER BY "embedding" <=> $1::vector
      LIMIT $2
    `,
    toVectorLiteral(jobEmbedding),
    limit
  )

  if (rows.length === 0) {
    return []
  }

  const profileIds = rows.map((row: VectorRow) => row.id)
  const profiles = await prisma.profile.findMany({
    where: {
      id: {
        in: profileIds
      }
    }
  })

  const profileById = new Map(profiles.map((profile: ProfileForEmbedding) => [profile.id, profile]))

  return profileIds
    .map((id: string) => profileById.get(id))
    .filter((profile: ProfileForEmbedding | undefined): profile is ProfileForEmbedding => Boolean(profile))
}
