"use client";

import React, { useEffect, useRef, useState } from "react";

/**
 * Car3D — photoreal, interactive 3D car (three.js, WebGL, HD pixel ratio).
 *
 * Install once:   npm i three
 *
 * Model: loads a real glTF car. Tries, in order:
 *   1. props.modelUrl           (optional, your own .glb)
 *   2. /models/car.glb          (drop any .glb into  public/models/car.glb)
 *   3. three.js "ferrari.glb"   (official three.js example asset, CC-BY, by vicent091036)
 * If every source fails, a procedural low-poly car is shown instead.
 *
 * Live props drive the car body: pitch ← accelX, roll ← accelY, shake on crash.
 */

const MODEL_SOURCES = (custom) =>
  [
    custom,
    "/models/car.glb",
    "https://threejs.org/examples/models/gltf/ferrari.glb",
    "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/ferrari.glb",
  ].filter(Boolean);

const STATUS_STYLE = {
  NORMAL: { body: 0xf4f6f8, glow: 0x000000, intensity: 0 },
  "RASH DRIVING": { body: 0xffc247, glow: 0xff8a00, intensity: 0.12 },
  "CRASH DETECTED": { body: 0xe11d48, glow: 0xff0033, intensity: 0.35 },
};

// Sensor pins. Positions are approximate mounting points (fractions of the model's bounding box x, y, z).
// Each pin shows its part name and a live reading that updates with the dashboard data.
const HOTSPOTS = [
  { id: "gnss", label: "GPS · NEO-8M", f: [0.5, 0.97, 0.42], color: "#0284c7", value: (L) => (L.speed == null ? "No GPS in recording" : `${Math.round(L.speed)} km/h`) },
  { id: "imu", label: "IMU · ISM330DHCX", f: [0.5, 0.35, 0.5], color: "#059669", value: (L) => `Ax ${L.accelX.toFixed(1)} · Ay ${L.accelY.toFixed(1)} m/s²` },
  { id: "cam", label: "Dashcam · PiCam", f: [0.5, 0.78, 0.66], color: "#7c3aed", value: (L) => (L.status === "CRASH DETECTED" ? "Saving 15 s clip" : "Recording · 15 fps") },
  { id: "buzz", label: "Buzzer · GPIO 18", f: [0.5, 0.55, 0.78], color: "#d97706", value: (L) => (L.status === "CRASH DETECTED" ? "Alarm" : L.status === "RASH DRIVING" ? "Beeping" : "Armed") },
];

export default function Car3D({
  status = "NORMAL",
  accelX = 0,
  accelY = 0,
  speed = 40,
  height = 360,
  autoRotate = true,
  showSensors = true,
  modelUrl,
  className = "",
}) {
  const mountRef = useRef(null);
  const labelRefs = useRef({});
  const valueRefs = useRef({});
  const live = useRef({ status, accelX, accelY, speed, autoRotate, showSensors });
  const [phase, setPhase] = useState("loading"); // loading | ready | fallback

  useEffect(() => {
    live.current = { status, accelX, accelY, speed, autoRotate, showSensors };
  }, [status, accelX, accelY, speed, autoRotate, showSensors]);

  useEffect(() => {
    let disposed = false;
    let raf = 0;
    let cleanup = () => {};

    (async () => {
      const THREE = await import("three");
      const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
      const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
      const { DRACOLoader } = await import("three/examples/jsm/loaders/DRACOLoader.js");
      const { RoomEnvironment } = await import("three/examples/jsm/environments/RoomEnvironment.js");
      if (disposed || !mountRef.current) return;

      const mount = mountRef.current;
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2.5)); // HD
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.05;
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      mount.appendChild(renderer.domElement);
      renderer.domElement.style.cssText = "width:100%;height:100%;display:block;touch-action:none;cursor:grab";

      const scene = new THREE.Scene();
      const pmrem = new THREE.PMREMGenerator(renderer);
      scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

      const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
      camera.position.set(4.6, 1.9, 5.4);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enablePan = false;
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.minDistance = 4;
      controls.maxDistance = 11;
      controls.maxPolarAngle = Math.PI / 2 - 0.04;
      controls.target.set(0, 0.5, 0);
      controls.autoRotateSpeed = 0.9;

      // Key light + soft ground shadow
      const sun = new THREE.DirectionalLight(0xffffff, 2.2);
      sun.position.set(4, 7, 5);
      sun.castShadow = true;
      sun.shadow.mapSize.set(2048, 2048);
      sun.shadow.camera.left = -6;
      sun.shadow.camera.right = 6;
      sun.shadow.camera.top = 6;
      sun.shadow.camera.bottom = -6;
      sun.shadow.bias = -0.0004;
      scene.add(sun);

      const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.28 }));
      floor.rotation.x = -Math.PI / 2;
      floor.receiveShadow = true;
      scene.add(floor);

      const ring = new THREE.Mesh(
        new THREE.RingGeometry(3.05, 3.12, 128),
        new THREE.MeshBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.45, side: THREE.DoubleSide })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.003;
      scene.add(ring);

      const rig = new THREE.Group(); // receives pitch / roll / shake
      scene.add(rig);

      const bodyMats = [];
      const wheels = [];
      let bbox = null;

      const finishModel = (model) => {
        model.traverse((o) => {
          if (o.isMesh) {
            o.castShadow = true;
            o.receiveShadow = true;
          }
        });
        // normalise size & ground it
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const k = 4.4 / Math.max(size.x, size.z);
        model.scale.setScalar(k);
        const box2 = new THREE.Box3().setFromObject(model);
        const c = box2.getCenter(new THREE.Vector3());
        model.position.set(-c.x, -box2.min.y, -c.z);
        rig.add(model);
        bbox = new THREE.Box3().setFromObject(model);
        rig.updateMatrixWorld(true);
      };

      const buildProcedural = () => {
        const g = new THREE.Group();
        const paint = new THREE.MeshPhysicalMaterial({ color: 0xf4f6f8, metalness: 0.7, roughness: 0.25, clearcoat: 1 });
        bodyMats.push(paint);
        const glass = new THREE.MeshPhysicalMaterial({ color: 0x0b1220, metalness: 0.9, roughness: 0.05 });
        const body = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.55, 4.3), paint);
        body.position.y = 0.62;
        const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.5, 2.1), glass);
        cabin.position.set(0, 1.15, -0.2);
        g.add(body, cabin);
        [[-0.95, 1.35], [0.95, 1.35], [-0.95, -1.35], [0.95, -1.35]].forEach(([x, z]) => {
          const w = new THREE.Mesh(
            new THREE.CylinderGeometry(0.42, 0.42, 0.3, 32),
            new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.6 })
          );
          w.rotation.z = Math.PI / 2;
          w.position.set(x, 0.42, z);
          g.add(w);
          wheels.push(w);
        });
        finishModel(g);
      };

      const draco = new DRACOLoader();
      draco.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.6/");
      const loader = new GLTFLoader();
      loader.setDRACOLoader(draco);

      const tryLoad = async (urls) => {
        for (const url of urls) {
          try {
            const gltf = await loader.loadAsync(url);
            if (disposed) return;
            const model = gltf.scene;
            const body = model.getObjectByName("body");
            if (body && body.material) {
              body.material = new THREE.MeshPhysicalMaterial({
                color: 0xf4f6f8,
                metalness: 1,
                roughness: 0.5,
                clearcoat: 1,
                clearcoatRoughness: 0.03,
              });
              bodyMats.push(body.material);
            }
            ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"].forEach((n) => {
              const w = model.getObjectByName(n);
              if (w) wheels.push(w);
            });
            const glass = model.getObjectByName("glass");
            if (glass && glass.material) {
              glass.material.metalness = 1;
              glass.material.roughness = 0;
              glass.material.opacity = 0.75;
              glass.material.transparent = true;
            }
            finishModel(model);
            setPhase("ready");
            return;
          } catch (e) {
            /* try next source */
          }
        }
        if (!disposed) {
          buildProcedural();
          setPhase("fallback");
        }
      };
      tryLoad(MODEL_SOURCES(modelUrl));

      const resize = () => {
        const w = mount.clientWidth || 600;
        const h = mount.clientHeight || 360;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };
      resize();
      const ro = new ResizeObserver(resize);
      ro.observe(mount);

      const clock = new THREE.Clock();
      const tmp = new THREE.Vector3();
      let pitch = 0;
      let roll = 0;
      const col = new THREE.Color();

      const loop = () => {
        raf = requestAnimationFrame(loop);
        const dt = Math.min(clock.getDelta(), 0.05);
        const t = clock.elapsedTime;
        const L = live.current;
        const st = STATUS_STYLE[L.status] || STATUS_STYLE.NORMAL;

        // body dynamics from IMU
        const targetPitch = THREE.MathUtils.clamp(-L.accelX * 0.0045, -0.09, 0.09);
        const targetRoll = THREE.MathUtils.clamp(L.accelY * 0.0055, -0.11, 0.11);
        pitch += (targetPitch - pitch) * Math.min(1, dt * 6);
        roll += (targetRoll - roll) * Math.min(1, dt * 6);
        rig.rotation.x = pitch;
        rig.rotation.z = roll;
        if (L.status === "CRASH DETECTED") {
          rig.position.x = Math.sin(t * 90) * 0.03;
          rig.position.y = Math.abs(Math.sin(t * 70)) * 0.025;
        } else {
          rig.position.x *= 0.8;
          rig.position.y *= 0.8;
        }

        // paint colour follows system state
        col.setHex(st.body);
        bodyMats.forEach((m) => {
          m.color.lerp(col, Math.min(1, dt * 5));
          if (m.emissive) {
            m.emissive.setHex(st.glow);
            m.emissiveIntensity = st.intensity * (0.7 + 0.3 * Math.sin(t * 8));
          }
        });

        // wheels
        const spin = L.status === "CRASH DETECTED" || L.speed == null ? 0 : (L.speed / 3.6) * dt * 1.6;
        wheels.forEach((w) => (w.rotation.x -= spin));

        controls.autoRotate = !!L.autoRotate;
        controls.update();
        ring.rotation.z += dt * 0.15;
        renderer.render(scene, camera);

        // project sensor hotspots to screen
        if (bbox) {
          const w = mount.clientWidth;
          const h = mount.clientHeight;
          HOTSPOTS.forEach((hs) => {
            const el = labelRefs.current[hs.id];
            if (!el) return;
            if (!L.showSensors) {
              el.style.opacity = 0;
              return;
            }
            const vEl = valueRefs.current[hs.id];
            if (vEl) {
              const txt = hs.value(L);
              if (vEl.textContent !== txt) vEl.textContent = txt;
            }
            tmp.set(
              bbox.min.x + (bbox.max.x - bbox.min.x) * hs.f[0],
              bbox.min.y + (bbox.max.y - bbox.min.y) * hs.f[1],
              bbox.min.z + (bbox.max.z - bbox.min.z) * hs.f[2]
            );
            rig.localToWorld(tmp);
            const camDist = tmp.distanceTo(camera.position);
            tmp.project(camera);
            const behind = tmp.z > 1;
            el.style.transform = `translate(${(tmp.x * 0.5 + 0.5) * w}px, ${(-tmp.y * 0.5 + 0.5) * h}px) translate(-50%, -50%)`;
            el.style.opacity = behind ? 0 : camDist > 9.5 ? 0.35 : 1;
          });
        }
      };
      loop();

      cleanup = () => {
        cancelAnimationFrame(raf);
        ro.disconnect();
        controls.dispose();
        draco.dispose();
        scene.traverse((o) => {
          if (o.geometry) o.geometry.dispose();
          if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
        });
        pmrem.dispose();
        renderer.dispose();
        if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      };
    })();

    return () => {
      disposed = true;
      cleanup();
    };
  }, [modelUrl]);

  return (
    <div
      className={`relative w-full overflow-hidden rounded-2xl ${className}`}
      style={{
        height,
        background:
          "radial-gradient(ellipse at 50% 75%, rgba(14,165,233,0.16), transparent 60%), linear-gradient(180deg, #f8fafc 0%, #e8f1f9 100%)",
      }}
    >
      <div ref={mountRef} className="absolute inset-0" />

      {/* Sensor hotspot labels */}
      <div className="absolute inset-0 pointer-events-none">
        {HOTSPOTS.map((hs) => (
          <div
            key={hs.id}
            ref={(el) => (labelRefs.current[hs.id] = el)}
            className="absolute left-0 top-0 flex items-center gap-1.5 transition-opacity duration-200"
            style={{ opacity: 0 }}
          >
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: hs.color }} />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full border border-white" style={{ background: hs.color }} />
            </span>
            <span
              className="rounded-md bg-white/90 px-1.5 py-0.5 shadow-sm border border-slate-200 whitespace-nowrap leading-tight"
              style={{ borderLeft: `3px solid ${hs.color}` }}
            >
              <span className="block text-[9px] font-bold text-slate-800">{hs.label}</span>
              <span ref={(el) => (valueRefs.current[hs.id] = el)} className="block text-[9px] font-semibold text-slate-500" />
            </span>
          </div>
        ))}
      </div>

      {phase === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-slate-500">
          Loading 3D model…
        </div>
      )}
      <div className="absolute bottom-2 left-3 text-[10px] font-semibold text-slate-400 pointer-events-none select-none">
        Pins = sensor positions (approx.), live values
      </div>
      <div className="absolute bottom-2 right-3 text-[10px] font-semibold text-slate-400 pointer-events-none select-none">
        Drag to rotate • Scroll to zoom
      </div>
      {phase === "fallback" && (
        <div className="absolute top-2 left-3 text-[10px] font-semibold text-amber-700 bg-amber-50 border border-amber-200 rounded-full px-2 py-0.5">
          Offline preview model — add public/models/car.glb for the HD car
        </div>
      )}
    </div>
  );
}
