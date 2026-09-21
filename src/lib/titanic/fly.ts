/**
 * Fly-connectome vision model, browser side.
 *
 * Passenger card -> 302 optic sensory neurons (retinotopic tiling) -> frozen
 * optic-lobe/central/descending wiring -> motor readout. The weights and the
 * trained parameters come from model.json.
 */

export type LayerSizes = number[];

interface RawModel {
	meta: {
		card_w: number;
		card_h: number;
		n_sensors: number;
		readout_size: number;
		layer_sizes: number[];
	};
	sensor_xy: number[][];
	sensor_gain: number[];
	inject: number[];
	weights: number[][];
	neuron_gain: number[][];
	neuron_bias: number[][];
	readout: number[];
	readout_bias: number;
	readout_idx: number[];
}

interface Model {
	meta: RawModel['meta'];
	sensor_xy: Array<[number, number]>;
	sensor_gain: Float32Array;
	inject: Float32Array;
	weights: Float32Array[];
	neuron_gain: Float32Array[];
	neuron_bias: Float32Array[];
	readout: Float32Array;
	readout_bias: number;
	readout_idx: number[];
}

let model: Model | null = null;

export function modelLoaded(): boolean {
	return !!model;
}

export function layerSizes(): LayerSizes | null {
	return model ? model.meta.layer_sizes : null;
}

export async function loadModel(base = '.'): Promise<Model> {
	const r = await fetch(`${base}/model.json`);
	const raw = (await r.json()) as RawModel;
	model = {
		meta: raw.meta,
		sensor_xy: raw.sensor_xy.map((p) => [p[0], p[1]] as [number, number]),
		sensor_gain: Float32Array.from(raw.sensor_gain),
		inject: Float32Array.from(raw.inject),
		weights: raw.weights.map((w) => Float32Array.from(w)),
		neuron_gain: raw.neuron_gain.map((g) => Float32Array.from(g)),
		neuron_bias: raw.neuron_bias.map((b) => Float32Array.from(b)),
		readout: Float32Array.from(raw.readout),
		readout_bias: raw.readout_bias,
		readout_idx: raw.readout_idx
	};
	return model;
}

export type Features = Record<string, number>;

export interface Prediction {
	img: Float32Array;
	sensors: Float32Array;
	layers: Float32Array[];
	logit: number;
	p: number;
}

/* ---- card rendering ----------------------------------------------------- */

const CW = 192;
const CH = 64;

function setRect(img: Float32Array, r0: number, r1: number, c0: number, c1: number, v: number) {
	for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) img[r * CW + c] = v;
}

function bar(img: Float32Array, r0: number, r1: number, frac: number) {
	const n = Math.round(CW * frac);
	for (let r = r0; r < r1; r++) for (let c = 0; c < n; c++) img[r * CW + c] = 1.0;
}

export function renderCard(f: Features): Float32Array {
	const img = new Float32Array(CW * CH);
	const clsV = f.pclass === 1 ? 0.9 : f.pclass === 2 ? 0.5 : 0.15;
	setRect(img, 0, 12, 0, CW, clsV);
	if (f.sex_male) setRect(img, 12, 24, 0, CW / 2, 0.9);
	else setRect(img, 12, 24, CW / 2, CW, 0.9);
	const age = Math.max(0, Math.min(80, f.age));
	bar(img, 24, 40, age / 80);
	const tick = Math.floor((28 / 80) * CW);
	setRect(img, 24, 40, tick, tick + 2, 1.0);
	const fare = Math.max(0, Math.min(300, f.fare));
	bar(img, 40, 50, Math.log1p(fare) / Math.log1p(300));
	bar(img, 50, 53, Math.min(8, f.sibsp) / 8);
	bar(img, 53, 56, Math.min(8, f.parch) / 8);
	const codes: Array<[number, string]> = [
		[0.0, 'title_Mr'],
		[0.08, 'title_Mrs'],
		[0.16, 'title_Miss'],
		[0.24, 'title_Master'],
		[0.32, 'title_Rare'],
		[0.45, 'deck_A'],
		[0.51, 'deck_B'],
		[0.57, 'deck_C'],
		[0.63, 'deck_D'],
		[0.69, 'deck_E'],
		[0.75, 'deck_F'],
		[0.83, 'embarked_C'],
		[0.88, 'embarked_Q'],
		[0.93, 'embarked_S']
	];
	for (const [frac, key] of codes) {
		if (f[key]) {
			const c0 = Math.floor(frac * CW);
			setRect(img, 57, 63, c0, c0 + 6, 0.9);
		}
	}
	return img;
}

/* ---- retinotopic tiling ------------------------------------------------ */

const GW = 8;
const GH = 38;

function clampInt(v: number, a: number, b: number): number {
	return v < a ? a : v > b ? b : v;
}

export function tileToSensors(img: Float32Array): Float32Array {
	if (!model) throw new Error('model not loaded');
	const n = model.sensor_xy.length;
	const gx = new Int32Array(n);
	const gy = new Int32Array(n);
	let xmin = Infinity,
		xmax = -Infinity,
		ymin = Infinity,
		ymax = -Infinity;
	for (const [x, y] of model.sensor_xy) {
		xmin = Math.min(xmin, x);
		xmax = Math.max(xmax, x);
		ymin = Math.min(ymin, y);
		ymax = Math.max(ymax, y);
	}
	for (let i = 0; i < n; i++) {
		const [x, y] = model.sensor_xy[i];
		gx[i] = clampInt(Math.round(((x - xmin) / (xmax - xmin + 1e-9)) * (GW - 1)), 0, GW - 1);
		gy[i] = clampInt(Math.round(((y - ymin) / (ymax - ymin + 1e-9)) * (GH - 1)), 0, GH - 1);
	}
	const patch = new Float32Array(GW * GH);
	for (let cx = 0; cx < GW; cx++)
		for (let cy = 0; cy < GH; cy++) {
			const c0 = Math.floor((cx / GW) * CW);
			const c1 = Math.max(c0 + 1, Math.floor(((cx + 1) / GW) * CW));
			const r0 = Math.floor((cy / GH) * CH);
			const r1 = Math.max(r0 + 1, Math.floor(((cy + 1) / GH) * CH));
			let s = 0;
			let k = 0;
			for (let r = r0; r < r1; r++)
				for (let c = c0; c < c1; c++) {
					s += img[r * CW + c];
					k++;
				}
			patch[cx * GH + cy] = k ? s / k : 0;
		}
	const acts = new Float32Array(n);
	for (let i = 0; i < n; i++) acts[i] = patch[gx[i] * GH + gy[i]];
	return acts;
}

/* ---- forward pass ------------------------------------------------------ */

function matVec(w: Float32Array, nIn: number, nOut: number, x: Float32Array): Float32Array {
	const y = new Float32Array(nOut);
	for (let j = 0; j < nOut; j++) {
		let s = 0;
		for (let i = 0; i < nIn; i++) s += x[i] * w[i * nOut + j];
		y[j] = s;
	}
	return y;
}

export function forward(sensors: Float32Array): {
	logit: number;
	p: number;
	layers: Float32Array[];
} {
	if (!model) throw new Error('model not loaded');
	const sizes = model.meta.layer_sizes;
	const sg = new Float32Array(sensors.length);
	for (let i = 0; i < sensors.length; i++) sg[i] = sensors[i] * model.sensor_gain[i];
	let a = matVec(model.inject, sensors.length, sizes[0], sg);
	const layers: Float32Array[] = [Float32Array.from(a)];
	for (let k = 0; k < model.weights.length; k++) {
		const z = matVec(model.weights[k], sizes[k], sizes[k + 1], a);
		const g = model.neuron_gain[k];
		const b = model.neuron_bias[k];
		const out = new Float32Array(sizes[k + 1]);
		for (let j = 0; j < sizes[k + 1]; j++) out[j] = Math.tanh(z[j] + b[j]) * g[j];
		a = out;
		layers.push(Float32Array.from(a));
	}
	let logit = model.readout_bias;
	for (let j = 0; j < model.readout.length; j++)
		logit += a[model.readout_idx[j]] * model.readout[j];
	const p = 1 / (1 + Math.exp(-logit));
	return { logit, p, layers };
}

/* ---- form -> prediction ------------------------------------------------ */

export interface FormValues {
	pclass: string;
	sex: string;
	title: string;
	embarked: string;
	age: string;
	fare: string;
	sibsp: string;
	parch: string;
	deck: string;
}

export function formToFeatures(v: FormValues): Features {
	const f: Features = {};
	f.pclass = Number(v.pclass);
	f.age = Number(v.age);
	f.fare = Number(v.fare);
	f.sibsp = Number(v.sibsp);
	f.parch = Number(v.parch);
	f.sex_male = v.sex === 'male' ? 1 : 0;
	const titles = ['title_Mr', 'title_Mrs', 'title_Miss', 'title_Master', 'title_Rare'];
	for (const t of titles) f[t] = v.title === t.replace('title_', '') ? 1 : 0;
	for (const d of 'ABCDEFG') f['deck_' + d] = v.deck === d ? 1 : 0;
	for (const e of ['C', 'Q', 'S']) f['embarked_' + e] = v.embarked === e ? 1 : 0;
	return f;
}

export function evaluate(features: Features): Prediction {
	const img = renderCard(features);
	const sensors = tileToSensors(img);
	const { logit, p, layers } = forward(sensors);
	return { img, sensors, layers, logit, p };
}
