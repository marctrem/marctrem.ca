/**
 * WebGL 3D brain-shaped activation view.
 *
 * Each neuron of the model's sub-circuit is drawn at its real FAFB (x, y, z)
 * position, coloured by its activation for the current passenger. Drag to
 * rotate, wheel to zoom. Raw WebGL, no dependencies.
 */

export interface BrainView {
	init: () => void;
	render: (points?: Float32Array) => void;
}

export function createBrain(canvas: HTMLCanvasElement): BrainView | null {
	const ctx =
		(canvas.getContext('webgl', {
			antialias: true,
			alpha: false
		}) as WebGLRenderingContext | null) ??
		(canvas.getContext('experimental-webgl') as WebGLRenderingContext | null);
	if (!ctx) return null;
	const gl: WebGLRenderingContext = ctx;

	const VS = `
		attribute vec3 aPos;
		attribute vec3 aCol;
		attribute float aSize;
		uniform mat4 uRot;
		uniform float uZoom;
		varying vec3 vCol;
		void main() {
			vCol = aCol;
			vec4 p = uRot * vec4(aPos, 1.0);
			gl_Position = vec4(p.xy * uZoom, 0.0, 1.0);
			gl_PointSize = aSize * uZoom;
		}`;
	const FS = `
		precision mediump float;
		varying vec3 vCol;
		void main() {
			vec2 d = gl_PointCoord - 0.5;
			if (dot(d, d) > 0.25) discard;
			gl_FragColor = vec4(vCol, 1.0);
		}`;

	function compile(src: string, type: number): WebGLShader {
		const s = gl.createShader(type) as WebGLShader;
		gl.shaderSource(s, src);
		gl.compileShader(s);
		if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? '');
		return s;
	}

	const prog = gl.createProgram() as WebGLProgram;
	gl.attachShader(prog, compile(VS, gl.VERTEX_SHADER));
	gl.attachShader(prog, compile(FS, gl.FRAGMENT_SHADER));
	gl.linkProgram(prog);
	const posAttr = gl.getAttribLocation(prog, 'aPos');
	const colAttr = gl.getAttribLocation(prog, 'aCol');
	const sizeAttr = gl.getAttribLocation(prog, 'aSize');
	const uRot = gl.getUniformLocation(prog, 'uRot');
	const uZoom = gl.getUniformLocation(prog, 'uZoom');
	const buf = gl.createBuffer();

	let yaw = -0.5;
	let pitch = 0.35;
	let zoom = 1;
	let dragging = false;
	let lastX = 0;
	let lastY = 0;
	let lastPoints: Float32Array | null = null;

	// rotation R = Ry(yaw) * Rx(pitch), column-major for WebGL
	function rotMat(y: number, p: number): Float32Array {
		const cy = Math.cos(y);
		const sy = Math.sin(y);
		const cp = Math.cos(p);
		const sp = Math.sin(p);
		return new Float32Array([
			cy,
			0,
			-sy,
			0,
			sy * sp,
			cp,
			cy * sp,
			0,
			sy * cp,
			-sp,
			cy * cp,
			0,
			0,
			0,
			0,
			1
		]);
	}

	function render(points?: Float32Array): void {
		if (points) lastPoints = points;
		if (!lastPoints) return;
		const n = lastPoints.length / 7;
		gl.viewport(0, 0, canvas.width, canvas.height);
		gl.clearColor(0.06, 0.07, 0.09, 1);
		gl.clear(gl.COLOR_BUFFER_BIT);
		gl.useProgram(prog);
		gl.bindBuffer(gl.ARRAY_BUFFER, buf);
		gl.bufferData(gl.ARRAY_BUFFER, lastPoints, gl.DYNAMIC_DRAW);
		const stride = 28;
		gl.enableVertexAttribArray(posAttr);
		gl.vertexAttribPointer(posAttr, 3, gl.FLOAT, false, stride, 0);
		gl.enableVertexAttribArray(colAttr);
		gl.vertexAttribPointer(colAttr, 3, gl.FLOAT, false, stride, 12);
		gl.enableVertexAttribArray(sizeAttr);
		gl.vertexAttribPointer(sizeAttr, 1, gl.FLOAT, false, stride, 24);
		gl.uniformMatrix4fv(uRot, false, rotMat(yaw, pitch));
		gl.uniform1f(uZoom, zoom);
		gl.drawArrays(gl.POINTS, 0, n);
	}

	function init(): void {
		canvas.style.cursor = 'grab';
		canvas.addEventListener('pointerdown', (e: PointerEvent) => {
			dragging = true;
			lastX = e.clientX;
			lastY = e.clientY;
			canvas.setPointerCapture(e.pointerId);
		});
		canvas.addEventListener('pointermove', (e: PointerEvent) => {
			if (!dragging) return;
			yaw += (e.clientX - lastX) * 0.01;
			pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - lastY) * 0.01));
			lastX = e.clientX;
			lastY = e.clientY;
			render();
		});
		canvas.addEventListener('pointerup', (e: PointerEvent) => {
			dragging = false;
			canvas.releasePointerCapture(e.pointerId);
		});
		canvas.addEventListener('pointercancel', () => {
			dragging = false;
		});
		canvas.addEventListener(
			'wheel',
			(e: WheelEvent) => {
				e.preventDefault();
				zoom = Math.max(0.4, Math.min(6, zoom * Math.exp(-e.deltaY * 0.001)));
				render();
			},
			{ passive: false }
		);
	}

	return { init, render };
}
