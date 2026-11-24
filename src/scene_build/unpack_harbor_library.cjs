// Convert glTF meshopt buffers and Basis textures into plain glTF for Blender 3.6.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const child = require('node:child_process');
const [directory, packages] = process.argv.slice(2);
const { MeshoptDecoder } = require(path.join(packages, 'meshoptimizer'));
const basis = path.join(packages, 'basis_universal/bin/basisu');

async function unpack(filename) {
  const raw = fs.readFileSync(path.join(directory, filename));
  const n = raw.readUInt32LE(12);
  const doc = JSON.parse(raw.subarray(20, 20 + n));
  if (raw.readUInt32LE(8) !== raw.length) throw Error('Incomplete ' + filename);
  const bin = raw.subarray(n + 28);
  const name = path.basename(filename, '.glb');
  const dest = path.join(directory, name);
  fs.mkdirSync(dest, { recursive: true });
  const parts = [];
  let cursor = 0;
  const bytes = [];
  for (const v of doc.bufferViews) {
    const e = v.extensions?.EXT_meshopt_compression;
    let b;
    if (e) {
      b = Buffer.alloc(e.count * e.byteStride);
      const offset = e.byteOffset || 0;
      MeshoptDecoder.decodeGltfBuffer(b, e.count, e.byteStride,
        bin.subarray(offset, offset + e.byteLength), e.mode, e.filter || 'NONE');
      delete v.extensions.EXT_meshopt_compression;
    } else {
      b = bin.subarray(v.byteOffset || 0, (v.byteOffset || 0) + v.byteLength);
    }
    bytes.push(b);
    v.buffer = 0;
    v.byteOffset = cursor;
    v.byteLength = b.length;
    const pad = (4 - b.length % 4) % 4;
    parts.push(b, Buffer.alloc(pad));
    cursor += b.length + pad;
  }
  for (const [i,img] of (doc.images || []).entries()) {
    const src = bytes[img.bufferView];
    let ext;
    if (img.mimeType === 'image/ktx2') {
      ext = 'png';
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(),'harbortex-'));
      try {
        fs.writeFileSync(path.join(tmp,'input.ktx2'),src);
        child.execFileSync(basis,['-unpack','-no_ktx','-format_only','3','input.ktx2'],{cwd:tmp,stdio:'ignore'});
        fs.copyFileSync(path.join(tmp,'input_unpacked_rgba_BC3_RGBA_0_0000.png'),path.join(dest,'t'+i+'.png'));
      } finally {
        fs.rmSync(tmp,{recursive:true,force:true});
      }
    } else {
      ext = img.mimeType === 'image/jpeg' ? 'jpg' : 'png';
      fs.writeFileSync(path.join(dest,'t'+i+'.'+ext),src);
    }
    img.uri='t'+i+'.'+ext;
    delete img.bufferView;
    delete img.mimeType;
  }
  for (const t of (doc.textures || [])) {
    if (t.extensions?.KHR_texture_basisu) {
      t.source=t.extensions.KHR_texture_basisu.source;
      delete t.extensions.KHR_texture_basisu;
    }
  }
  for (const key of ['extensionsUsed','extensionsRequired']) {
    if (doc[key]) doc[key]=doc[key].filter(x=>!['EXT_meshopt_compression','KHR_texture_basisu'].includes(x));
  }
  for (const [i,m] of (doc.materials || []).entries()) m.name='Slot'+String(i).padStart(2,'0');
  doc.buffers=[{byteLength:cursor,uri:'mesh.bin'}];
  fs.writeFileSync(path.join(dest,'mesh.bin'),Buffer.concat(parts));
  fs.writeFileSync(path.join(dest,'model.gltf'),JSON.stringify(doc));
  console.log(name);
}
(async () => {
  await MeshoptDecoder.ready;
  for (const f of fs.readdirSync(directory).filter(x=>x.endsWith('.glb'))) await unpack(f);
})().catch(e=>{console.error(e);process.exitCode=1;});
