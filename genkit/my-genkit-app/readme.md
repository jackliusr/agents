# Get started with Genkit

This guide shows you how to get started with Genkit in your preferred language and test it in the Developer UI.

## Prerequisites

Before you begin, make sure your environment meets these requirements:

- Node.js v20 or later
- pnpm

This guide assumes you're already familiar with building Node.js applications.

## Set up your project

Create a new Node.js project and configure TypeScript:

```bash
# Create and navigate to a new directory
mkdir my-genkit-app
cd my-genkit-app

# Initialize a new Node.js project
pnpm init
pnpm pkg set type=module

# Install and configure TypeScript
pnpm install -D typescript tsx
pnpm --package=typescript dlx tsc --init

# Set up your source directory
mkdir src
touch src/index.ts
```

This sets up your project structure and a TypeScript entry point at `src/index.ts`.

## Install Genkit packages

First, install the Genkit CLI globally. This gives you access to local developer tools, including the Developer UI:

```bash
pnpm install -g genkit-cli
```

Then, add the following packages to your project:

```bash
pnpm install genkit @genkit-ai/google-genai dotenv
```

- `genkit` provides Genkit core capabilities.
- `@genkit-ai/google-genai` provides access to the Google AI Gemini models.

## Configure your model API key

Genkit can work with multiple model providers. This guide uses the **Gemini API**, which offers a generous free tier and doesn't require a credit card to get started.

To use it, you'll need an API key from Google AI Studio:

<LinkButton
  href="https://aistudio.google.com/apikey"
  icon="external"
  target="_blank"
  rel="noopener noreferrer"
>
  Get a Gemini API Key
</LinkButton>

Once you have a key, set the `GEMINI_API_KEY` environment variable in .env file:

```
GEMINI_API_KEY=
```

:::note
Genkit also supports models from Vertex AI, Anthropic, OpenAI, Cohere, Ollama, and more. See [generating content](/docs/js/models/) for details.
:::

## Create your first application

A flow is a special Genkit function with built-in observability, type safety, and tooling integration.

Update `src/index.ts` with the following:

```ts
import { googleAI } from '@genkit-ai/google-genai';
import { genkit, z } from 'genkit';

// Initialize Genkit with the Google AI plugin
const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-2.5-flash', {
    temperature: 0.8,
  }),
});

// Define input schema
const RecipeInputSchema = z.object({
  ingredient: z.string().describe('Main ingredient or cuisine type'),
  dietaryRestrictions: z
    .string()
    .optional()
    .describe('Any dietary restrictions'),
});

// Define output schema
const RecipeSchema = z.object({
  title: z.string(),
  description: z.string(),
  prepTime: z.string(),
  cookTime: z.string(),
  servings: z.number(),
  ingredients: z.array(z.string()),
  instructions: z.array(z.string()),
  tips: z.array(z.string()).optional(),
});

// Define a recipe generator flow
export const recipeGeneratorFlow = ai.defineFlow(
  {
    name: 'recipeGeneratorFlow',
    inputSchema: RecipeInputSchema,
    outputSchema: RecipeSchema,
  },
  async (input) => {
    // Create a prompt based on the input
    const prompt = `Create a recipe with the following requirements:
      Main ingredient: ${input.ingredient}
      Dietary restrictions: ${input.dietaryRestrictions || 'none'}`;

    // Generate structured recipe data using the same schema
    const { output } = await ai.generate({
      prompt,
      output: { schema: RecipeSchema },
    });

    if (!output) throw new Error('Failed to generate recipe');

    return output;
  },
);

// Run the flow
async function main() {
  const recipe = await recipeGeneratorFlow({
    ingredient: 'avocado',
    dietaryRestrictions: 'vegetarian',
  });

  console.log(recipe);
}

main().catch(console.error);
```

This code sample:

- Defines reusable input and output schemas with [Zod](https://zod.dev/)
- Configures the `gemini-2.5-flash` model with temperature settings
- Defines a Genkit flow to generate a structured recipe based on your input
- Runs the flow with a sample input and prints the result

### Why use flows?

- **Type-safe inputs and outputs**: Define clear schemas for your data
- **Integrates with the Developer UI**: Test and debug flows visually
- **Easy deployment as APIs**: Deploy flows as HTTP endpoints
- **Built-in tracing and observability**: Monitor performance and debug issues

## Run your application

Run your application to see it in action:

```bash
npx tsx src/index.ts
```

You should see a structured recipe output in your console.

## Test in the Developer UI

The **Developer UI** is a local tool for testing and inspecting Genkit components, like flows, with a visual interface.

### Start the Developer UI

The Genkit CLI is required to run the Developer UI. If you followed the installation steps above, you already have it installed.

Run the following command from your project root:

```bash
genkit start -- npx tsx --watch src/index.ts
```

This starts your app and launches the Developer UI at `http://localhost:4000` by default.

:::note
The command after `--` should run the file that defines or imports your Genkit components. You can use `tsx`, `node`, or other commands based on your setup. Learn more in [developer tools](/docs/js/devtools/).
:::

##### Optional: Add an npm script

To make starting the Developer UI easier, add the following to your `package.json` scripts:

```json
"scripts": {
  "genkit:ui": "genkit start -- npx tsx --watch src/index.ts"
}
```

Then run it with:

```bash
npm run genkit:ui
```

### Run and inspect flows

In the Developer UI:

1. Select `recipeGeneratorFlow` from the list of flows.

2. Enter sample input:

   

   ```json
   {
     "ingredient": "avocado",
     "dietaryRestrictions": "vegetarian"
   }
   ```

   

   

3. Click **Run**

   You'll see the generated recipe as structured output, along with a visual trace of the AI generation process for debugging and optimization.

   <video
     controls
     preload="metadata"
     muted
     width="100%"
     style="max-width: 800px; display: block; margin: 2rem auto 0; border: none;"
     poster="/videos/devui-flow-runner-poster.png"
     aria-label="Demonstration of running a Genkit flow in the Developer UI, showing the recipe generator flow with input parameters and structured output results"
   >
     <source src="/videos/devui-flow-runner.mp4" type="video/mp4" />
     Your browser does not support the video tag. This video demonstrates how to
     run a Genkit flow in the Developer UI, showing the recipe generator flow
     with input parameters and the resulting structured output.
   </video>

## Next steps

Now that you've created and tested your first Genkit application, explore more features to build powerful AI-driven applications:

- [Developer tools](/docs/js/devtools/): Set up your local workflow with the Genkit CLI and Dev UI.
- [Generating content](/docs/js/models/): Use Genkit's unified generation API to work with multimodal and structured output across supported models.
- [Creating flows](/docs/js/flows/): Learn about streaming flows, schema customization, deployment options, and more.
- [Tool calling](/docs/js/tool-calling/): Enable your AI models to interact with external systems and APIs.
- [Managing prompts with Dotprompt](/docs/js/dotprompt/): Define flexible prompt templates using `.prompt` files or code.