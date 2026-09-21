<script lang="ts">
	import { resolve } from '$app/paths';
</script>

<svelte:head>
	<title>Marc Tremblay</title>
	<meta name="description" content="Personal homepage of Marc Tremblay." />
</svelte:head>

<main>
	<h1>Marc Tremblay</h1>
	<p class="tagline">Personal homepage</p>
	<nav>
		<a href="https://github.com/marctrem">GitHub</a>
		<a href={resolve('/titanic-fly-classifier')}>Titanic fly classifier</a>
		<span class="email">m&nbsp;[@]&nbsp;rctrem.ca</span>
	</nav>
</main>
