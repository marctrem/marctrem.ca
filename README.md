# marctrem.ca

Personal homepage. [SvelteKit](https://svelte.dev/docs/kit) (static, `@sveltejs/adapter-static`) hosted on [Cloudflare Pages](https://pages.cloudflare.com/).

## Develop

```sh
npm install
npm run dev -- --open
```

## Build & preview

```sh
npm run build
npm run preview
```

## Deploy (Cloudflare Pages)

```sh
npx wrangler login
npx wrangler pages deploy build
```

Or connect this repository in the Cloudflare dashboard: build command `npm run build`, output directory `build` (see `wrangler.jsonc`). The custom domain `marctrem.ca` can be attached to the Pages project.
