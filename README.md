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

Pushes to `main` deploy automatically via the Cloudflare Pages GitHub integration (project `marctrem-ca`: build command `npm run build`, output directory `build`).

For one-off manual deploys:

```sh
npx wrangler login
npx wrangler pages deploy build
```
