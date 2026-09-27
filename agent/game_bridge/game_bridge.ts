/**
 * Game Bridge for Phase 5 - Clean Implementation with Working Attacks
 *
 * Features:
 * - Battle royale mode with configurable bots
 * - Working attack execution using AttackExecution
 * - Spatial map extraction (512×512 → 128×128)
 * - 16 global features
 * - GPU-optimized
 */

import * as readline from 'readline';
import { PlayerType, Player, Game, PlayerInfo, GameMapType, GameMode, GameType, Difficulty } from '../../base-game/src/core/game/Game';
import { createGame } from '../../base-game/src/core/game/GameImpl';
import { genTerrainFromBin, MapManifest } from '../../base-game/src/core/game/TerrainMapLoader';
import { UserSettings } from '../../base-game/src/core/game/UserSettings';
import { GameConfig } from '../../base-game/src/core/Schemas';
import { TestConfig } from '../../base-game/tests/util/TestConfig';
import { DefaultConfig } from '../../base-game/src/core/configuration/DefaultConfig';
import { TestServerConfig } from '../../base-game/tests/util/TestServerConfig';
import { AttackExecution } from '../../base-game/src/core/execution/AttackExecution';
import { SpawnExecution } from '../../base-game/src/core/execution/SpawnExecution';
import * as path from 'path';
import * as fs from 'fs';
import { dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface TerritoryCluster {
  id: number;
  tiles: number[];
  tile_count: number;   // SKAI: size sans matérialiser les index (JSON ~10x plus léger)
  border_tiles: number[];
  center_x: number;
  center_y: number;
  troop_count: number;
}

interface GameState {
  // Core state
  tick: number;
  game_over: boolean;
  has_won: boolean;
  has_lost: boolean;

  // Territory info
  tiles_owned: number;
  total_tiles: number;
  territory_pct: number;        // NEW: fraction of LAND tiles owned (the only meaningful ratio)
  territory_pct_all: number;    // legacy: fraction of the whole rectangle (water included)
  land_tiles: number;           // NEW: number of land tiles on the map
  neutral_tiles: number;
  water_mask: number[][];       // NEW: 64x64 land fraction per block (1 = land, 0 = ocean)

  // Population
  population: number;
  max_population: number;
  population_growth_rate: number;

  // Resources
  gold: number;
  num_cities: number;

  // Position
  rank: number;
  total_players: number;
  alive_players: number;

  // Spatial data (512×512 or map size)
  territory_map: number[][];
  troop_map: number[][];

  // Global features for RL
  border_tiles: number;
  border_pressure: number;
  time_alive: number;
  nearest_threat_distance: number;
  territory_change: number;

  // NEW: Territory clusters (for disconnected territories)
  clusters: TerritoryCluster[];

  // NEW: Economic & Military features
  cities_count: number;
  ports_count: number;
  silos_count: number;
  sam_launchers_count: number;
  defense_posts_count: number;
  factories_count: number;
  atom_bombs_available: number;
  hydrogen_bombs_available: number;
  can_build_city: boolean;
  can_build_port: boolean;
  can_build_silo: boolean;
  can_build_sam: boolean;
  can_launch_nuke: boolean;

  // NEW: Unit positions for spatial observations
  our_cities_positions: Array<{x: number, y: number}>;
  our_ports_positions: Array<{x: number, y: number}>;
  our_silos_positions: Array<{x: number, y: number}>;
  our_sam_positions: Array<{x: number, y: number}>;
  our_defense_positions: Array<{x: number, y: number}>;
  our_factories_positions: Array<{x: number, y: number}>;
  enemy_buildings_positions: Array<{x: number, y: number}>;
}

interface Command {
  type: 'reset' | 'tick' | 'get_state' | 'attack_direction' | 'cancel_attacks' | 'build_unit' | 'launch_nuke' | 'shutdown';
  map_name?: string;
  num_players?: number;
  cluster_id?: number;   // NEW: which cluster to control (0-4)
  direction?: number;    // 0-8 (N, NE, E, SE, S, SW, W, NW, WAIT)
  intensity?: number;    // 0.0-1.0 (fraction of the RESERVE committed to the attack)
  source_tile?: boolean; // NEW: true => attack spreads from the chosen border tile only (real direction)
  win_threshold?: number; // NEW: share of LAND tiles that counts as a win
  verbose?: boolean;      // NEW: per-attack logs on stderr (for the visualizer / debugging)
  game_config?: string;   // NEW: 'default' (real rules) | 'test' (legacy stub config)
  max_attacks?: number;   // NEW: concurrent attack fronts allowed (default 4)
  spawn_mode?: string;    // NEW: 'land' (défaut) | 'ring' (ancien comportement)
  spawn_seed?: number;    // NEW: graine de placement (diversifie les départs entre épisodes)
  obs_size?: number;      // NEW: downsample territory_map to this square size (0 = full res)
  allow_multi_front?: boolean; // NEW: autorise plusieurs fronts simultanés vs le même adversaire
                              // (ratio-lab). Désactivé par défaut pour l'entraînement RL.
  // NEW: Building commands
  unit_type?: string;    // 'City', 'Port', 'Missile Silo', 'SAM Launcher', 'Defense Post'
  tile_x?: number;       // Normalized 0-1 position for building/nuke target
  tile_y?: number;
  nuke_type?: string;    // 'Atom Bomb' or 'Hydrogen Bomb'
}

class GameBridge {
  private game: Game | null = null;
  private rlPlayer: Player | null = null;
  private aiPlayers: Player[] = [];
  private currentTick: number = 0;
  private maxTicks: number = 50000;

  // Map cache
  private mapName: string = 'plains';
  private numPlayers: number = 11;  // 1 RL + 10 bots
  private mapWidth: number = 512;
  private mapHeight: number = 512;

  // History for computing rates
  private previousTiles: number = 0;
  private territoryHistory: number[] = [];

  // Map geometry, computed once at init (see SKAI patches)
  private landTiles: number = 0;
  private waterMask: number[][] = [];
  // SKAI: static land bitmap (1 = land), built once per map, so the per-tick scan
  // does not need game.isLand()/game.owner() (6 nested calls per tile).
  private landBits: Uint8Array = new Uint8Array(0);
  // SKAI: own-territory bitmap from the fused scan (0/1 per tile), + cluster memo
  private scanOwn: Uint8Array = new Uint8Array(0);
  private clustersCacheTick: number = -1;
  private clustersCache: TerritoryCluster[] = [];
  // SKAI: per-player centroids, refreshed by the single fused map scan each tick
  private centers: Map<number, { x: number; y: number; n: number }> = new Map();
  // SKAI: share of LAND needed to declare a win (was 80% of the whole rectangle)
  private winThreshold: number = 0.80;
  // SKAI: per-episode bridge logs (off during training, on for the visualizer)
  private verbose: boolean = false;
  // SKAI: 'default' = real OpenFront rules, 'test' = legacy TestConfig stubs
  private gameConfigMode: 'default' | 'test' = 'default';
  // SKAI: cap on concurrent attack fronts
  private maxAttacks: number = 4;
  // SKAI: 'land' = répartition sur terres séparées, 'ring' = comportement historique
  private spawnMode: 'land' | 'ring' = 'land';
  // SKAI: graine de placement des spawners (variable à chaque épisode d'entraînement)
  private spawnSeed: number = 12345;
  // SKAI: if > 0, territory_map is returned downsampled to obs_size x obs_size.
  // Sending a full 500x500 map as JSON text costs ~750 kB per tick, which made the
  // bridge crawl at ~12 ticks/s (measured) - the serialisation, not the simulation.
  private obsSize: number = 0;
  // SKAI (ratio-lab): when true, the "front already carries enough troops" guard is
  // skipped so a sweep bot may hold several simultaneous fronts against the same
  // target (e.g. terra nullius). Off by default: RL training relies on the guard.
  private allowMultiFront: boolean = false;

  constructor() {
    this.log('GameBridge Phase 5 initialized (Clean with working attacks)');
  }

  /**
   * Initialize/reset game
   */
  async initialize(mapName: string = 'plains', numPlayers: number = 11): Promise<GameState> {
    this.mapName = mapName;
    this.numPlayers = numPlayers;

    try {
      // Create player list: 1 RL agent + (numPlayers-1) AI bots
      const players: PlayerInfo[] = [
        new PlayerInfo('RL_Agent', PlayerType.Human, null, 'RL_Agent')
      ];

      // Add AI bots
      for (let i = 1; i < numPlayers; i++) {
        players.push(
          new PlayerInfo(`AI_Bot_${i}`, PlayerType.Bot, null, `AI_Bot_${i}`)
        );
      }

      // Load map
      const mapDir = path.join(__dirname, '../../base-game/resources/maps', mapName);
      const manifestPath = path.join(mapDir, 'manifest.json');

      if (!fs.existsSync(manifestPath)) {
        throw new Error(`Map not found: ${mapName} at ${manifestPath}`);
      }

      const manifest: MapManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
      this.mapWidth = manifest.map.width;
      this.mapHeight = manifest.map.height;

      // Load binary map data (both main map and mini map)
      const mapBinPath = path.join(mapDir, 'map.bin');
      const miniMapBinPath = path.join(mapDir, 'mini_map.bin');
      const mapBinData = fs.readFileSync(mapBinPath);
      const miniMapBinData = fs.readFileSync(miniMapBinPath);

      // Generate terrain from binary data
      const gameMap = await genTerrainFromBin(manifest.map, mapBinData);
      const miniGameMap = await genTerrainFromBin(manifest.mini_map, miniMapBinData);

      // ---- SKAI: cache map geometry once ------------------------------------
      // territory_pct MUST be measured against land tiles: the engine never lets
      // a player own ocean tiles, so a denominator of width*height makes any win
      // threshold unreachable on maps with sea (and hides the real progress).
      this.landTiles = Math.max(1, manifest.map.num_land_tiles);
      this.waterMask = this.buildWaterMask(gameMap, this.mapWidth, this.mapHeight, 64);
      this.landBits = new Uint8Array(this.mapWidth * this.mapHeight);
      for (let y = 0; y < this.mapHeight; y++) {
        for (let x = 0; x < this.mapWidth; x++) {
          if (gameMap.isLand(gameMap.ref(x, y))) this.landBits[y * this.mapWidth + x] = 1;
        }
      }
      const landFrac = (this.landTiles / (this.mapWidth * this.mapHeight)) * 100;
      this.log(`Map ${mapName}: ${this.mapWidth}x${this.mapHeight}, land tiles=${this.landTiles} (${landFrac.toFixed(1)}% of the rectangle)`);
      if (this.landTiles <= 1) {
        this.log(`WARNING: map '${mapName}' has NO land tile -> unusable for training, players spawn on ocean.`);
      }

      // Configure the game
      const gameConfig: GameConfig = {
        // SKAI: was hardcoded to GameMapType.Asia whatever the loaded file; the value is
        // metadata (config/reporting), so keep it plausible for the map actually in memory.
        gameMap: mapName.toLowerCase().includes('australia') ? GameMapType.Australia : GameMapType.Asia,
        gameMode: GameMode.FFA,
        gameType: GameType.Singleplayer,
        difficulty: Difficulty.Easy,
        disableNPCs: false,
        donateGold: false,
        donateTroops: false,
        bots: 0,
        infiniteGold: false,
        infiniteTroops: false,
        instantBuild: false,
      };

      // Suppress console.debug (reduces noise)
      console.debug = () => {};

      // ---- SKAI: config selection -------------------------------------------
      // The bridge used to hard-wire `TestConfig`, the *Jest unit-test* config.
      // That config pins the combat rules to deterministic stubs:
      //   attackTilesPerTick() -> 1        (real DefaultConfig: numAdjacent*2 vs neutral,
      //                                     up to ~0.5*numAdjacent*3 vs a player)
      //   attackLogic()        -> 1 troop / 1 tile / 1 tile per tick
      //   samHittingChance()   -> 1
      //   nukeMagnitudes()     -> {inner:1, outer:1}
      // i.e. ONE tile conquered per tick per attack, forever: on a 100x100 map an
      // episode could never exceed ~15% territory, so the "80% = win" bonus never
      // fired and the reward was a constant drift the policy could not influence.
      // Training must use DefaultConfig, which is what the real game plays with.
      const serverConfig = new TestServerConfig();
      const config =
        this.gameConfigMode === 'test'
          ? new TestConfig(serverConfig, gameConfig, new UserSettings(), false)
          : new DefaultConfig(serverConfig, gameConfig, new UserSettings(), false);
      this.log(`Game config: ${this.gameConfigMode} (DefaultConfig = real combat rules)`);

      // Create game with positional arguments (not an object!)
      this.game = createGame(players, [], gameMap, miniGameMap, config);

      if (!this.game) {
        throw new Error('createGame() returned null');
      }

      this.log(`Game created, spawning ${numPlayers} players...`);

      // Spawn players
      // SKAI: the old placement put everyone on a circle of radius min(w,h)*0.4,
      // ignoring terrain: on a real map (australia_real_*, 1024x1024) the ring crosses
      // the sea, several players snapped onto the SAME land patch, and with 400 tribes
      // the ring is simply not a map of Australia any more. We now sample well-separated
      // LAND tiles (Poisson-disk on a grid), which is also closer to how the real game
      // places bots. `spawn_mode: "ring"` restores the legacy behaviour.
      const spawnExecutions: SpawnExecution[] = [];
      const spawnPoints =
        this.spawnMode === 'ring'
          ? this.ringSpawns(players.length)
          : this.landSpawns(players.length);

      for (let i = 0; i < players.length; i++) {
        const spawnPos = spawnPoints[i] ?? this.findLandNear(this.mapWidth / 2, this.mapHeight / 2);
        spawnExecutions.push(new SpawnExecution(players[i], spawnPos));
      }

      this.game.addExecution(...spawnExecutions);

      // Execute spawn phase
      while (this.game.inSpawnPhase()) {
        this.game.executeNextTick();
      }

      // Get players
      this.rlPlayer = this.game.player('RL_Agent');
      if (!this.rlPlayer) {
        throw new Error('Could not find RL_Agent');
      }

      this.aiPlayers = [];
      for (let i = 1; i < numPlayers; i++) {
        const aiPlayer = this.game.player(`AI_Bot_${i}`);
        if (aiPlayer) {
          this.aiPlayers.push(aiPlayer);
        }
      }

      this.currentTick = 0;
      this.previousTiles = this.rlPlayer.numTilesOwned();
      this.territoryHistory = [];

      // Verify clusters are formed
      const initialClusters = this.detectTerritoryClusters();
      this.log(`Game initialized: map=${mapName}, size=${this.mapWidth}x${this.mapHeight}, players=${numPlayers}`);
      this.log(`RL Agent starting with ${this.rlPlayer.numTilesOwned()} tiles, ${initialClusters.length} cluster(s)`);

      return this.getState();
    } catch (error: any) {
      this.log(`Initialization error: ${error}`);
      this.log(`Stack trace: ${error.stack}`);
      throw error;
    }
  }

  /**
   * SKAI: block-majority downsample of the territory map (keeps the owner that covers
   * the most tiles in each block, so coastlines and fronts survive the shrink).
   */
  private downsampleMap(map: number[][], size: number): number[][] {
    const h = this.mapHeight, w = this.mapWidth;
    if (h <= size && w <= size) return map;
    const out: number[][] = [];
    for (let by = 0; by < size; by++) {
      const y0 = Math.floor((by * h) / size);
      const y1 = Math.max(y0 + 1, Math.floor(((by + 1) * h) / size));
      const row: number[] = new Array(size);
      for (let bx = 0; bx < size; bx++) {
        const x0 = Math.floor((bx * w) / size);
        const x1 = Math.max(x0 + 1, Math.floor(((bx + 1) * w) / size));
        const counts = new Map<number, number>();
        let best = 0, bestN = 0;
        for (let y = y0; y < y1; y++) {
          const r = map[y];
          for (let x = x0; x < x1; x++) {
            const v = r[x];
            const n = (counts.get(v) || 0) + 1;
            counts.set(v, n);
            if (n > bestN) { bestN = n; best = v; }
          }
        }
        row[bx] = best;
      }
      out.push(row);
    }
    return out;
  }

  /** legacy behaviour: evenly spaced points on a circle, snapped to land */
  private ringSpawns(count: number): number[] {
    const out: number[] = [];
    const step = (2 * Math.PI) / Math.max(count, 1);
    const radius = Math.min(this.mapWidth, this.mapHeight) * 0.4;
    for (let i = 0; i < count; i++) {
      const x = Math.floor(this.mapWidth / 2 + Math.cos(i * step) * radius);
      const y = Math.floor(this.mapHeight / 2 + Math.sin(i * step) * radius);
      out.push(this.findLandNear(x, y));
    }
    return out;
  }

  /**
   * SKAI: Poisson-disk sampling over land tiles (grid-accelerated).
   * spacing = sqrt(land_area / count) so that `count` players spread evenly over the
   * continent; if the map cannot honour it, the radius shrinks by 0.7 each round.
   */
  private landSpawns(count: number): number[] {
    const w = this.mapWidth, h = this.mapHeight;
    const land = this.landBits;
    const isReal = this.landTiles > 1;
    const candidates: number[] = [];
    const stride = Math.max(1, Math.floor(Math.sqrt((isReal ? this.landTiles : w * h) / Math.max(count, 1)) / 2));
    for (let i = 0; i < land.length; i++) {
      if (land[i] && (i % stride === 0 || stride <= 1)) candidates.push(i);
    }
    if (candidates.length < count) {
      candidates.length = 0;
      for (let i = 0; i < land.length; i++) if (land[i]) candidates.push(i);
    }
    // SKAI: melange déterministe mais piloté par `spawn_seed`. Sans ça, le parcours
    // ligne-par-ligne plaçait le joueur 0 tout au nord de l'Australie à CHAQUE
    // épisode : un agent RL mémorise alors une position au lieu d'apprendre un jeu.
    let rngState = (this.spawnSeed >>> 0) || 1;
    const rnd = () => {
      rngState ^= rngState << 13; rngState >>>= 0;
      rngState ^= rngState >> 17;
      rngState ^= rngState << 5; rngState >>>= 0;
      return rngState / 4294967296;
    };
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const tmp = candidates[i]; candidates[i] = candidates[j]; candidates[j] = tmp;
    }

    let radius = Math.max(3, Math.sqrt(this.landTiles / Math.max(count, 1)));
    for (let attempt = 0; attempt < 6 && radius > 1; attempt++) {
      const cell = Math.max(1, Math.floor(radius));
      const gw = Math.ceil(w / cell), gh = Math.ceil(h / cell);
      const grid: number[][] = new Array(gw * gh);
      const chosen: number[] = [];
      for (const t of candidates) {
        const x = t % w, y = (t - x) / w;
        const gx = Math.floor(x / cell), gy = Math.floor(y / cell);
        let ok = true;
        for (let oy = -1; oy <= 1 && ok; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            const nx = gx + ox, ny = gy + oy;
            if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
            const bucket = grid[ny * gw + nx];
            if (!bucket) continue;
            for (const o of bucket) {
              const oxx = o % w, oyy = (o - oxx) / w;
              if ((oxx - x) * (oxx - x) + (oyy - y) * (oyy - y) < radius * radius) { ok = false; break; }
            }
            if (!ok) break;
          }
        }
        if (!ok) continue;
        const key = gy * gw + gx;
        (grid[key] || (grid[key] = [])).push(t);
        chosen.push(t);
        if (chosen.length >= count) break;
      }
      if (chosen.length >= count) return chosen;
      // pas assez de places à cette distance : on resserre
      const filled = chosen;
      if (filled.length > 0 && attempt === 5) return filled;
      radius *= 0.7;
    }
    return candidates.slice(0, count);
  }

  /**
   * SKAI: nearest land tile to (x,y), searched in growing rings. Falls back to the
   * centre of the map if the whole map is water (broken map file).
   */
  private findLandNear(x: number, y: number): number {
    const game = this.game!;
    const clampX = Math.max(1, Math.min(this.mapWidth - 2, Math.floor(x)));
    const clampY = Math.max(1, Math.min(this.mapHeight - 2, Math.floor(y)));
    for (let r = 0; r < Math.max(this.mapWidth, this.mapHeight); r++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
          const nx = clampX + dx, ny = clampY + dy;
          if (nx < 0 || ny < 0 || nx >= this.mapWidth || ny >= this.mapHeight) continue;
          const t = game.ref(nx, ny);
          if (game.isLand(t)) return t;
        }
      }
    }
    return game.ref(Math.floor(this.mapWidth / 2), Math.floor(this.mapHeight / 2));
  }

  /**
   * Execute one game tick
   */
  tick(): GameState {
    if (!this.game) throw new Error('Game not initialized');

    this.game.executeNextTick();
    this.currentTick++;

    return this.getState();
  }

  /**
   * Get current game state with spatial maps and global features
   */
  getState(): GameState {
    if (!this.game || !this.rlPlayer) {
      throw new Error('Game not initialized');
    }

    const aliveAIPlayers = this.aiPlayers.filter(ai => ai.isAlive());
    const alivePlayers = [this.rlPlayer, ...aliveAIPlayers].filter(p => p.isAlive());

    // Territory info
    const tilesOwned = this.rlPlayer.numTilesOwned();
    const totalTiles = this.mapWidth * this.mapHeight;
    // SKAI: the progress signal is the share of LAND owned, not of the rectangle.
    const territoryPct = tilesOwned / this.landTiles;
    const territoryPctAll = tilesOwned / totalTiles;

    // SKAI: single fused pass over the map -> neutral-land count + player centroids.
    // (Was 1 pass for neutral tiles + 1 pass per enemy for "nearest threat" + 1 pass
    //  for the territory map: O(w*h*(2+n)) per tick, which dominated episode time.)
    const scan = this.scanMap();
    const neutralTiles = scan.neutralLand;
    this.centers = scan.centers;
    // SKAI: clusters are derived from this same scan instead of a second full
    // flood-fill pass that ran once per getState AND once per attack command.
    this.scanOwn = scan.own;
    this.clustersCacheTick = -1;

    // Population
    const population = this.rlPlayer.troops();
    const maxPopulation = this.game.config().maxTroops(this.rlPlayer);
    const populationGrowthRate = (tilesOwned - this.previousTiles) / Math.max(this.previousTiles, 1);

    // Resources
    const gold = Number(this.rlPlayer.gold());
    const numCities = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'City').length;

    // Rank
    const alivePlayers_copy = alivePlayers.slice().sort((a, b) => b.numTilesOwned() - a.numTilesOwned());
    const rank = alivePlayers_copy.findIndex(p => p.id() === this.rlPlayer.id()) + 1;

    // Spatial maps (already produced by the fused scan)
    const territoryMap = this.obsSize > 0 ? this.downsampleMap(scan.map, this.obsSize) : scan.map;
    const troopMap: number[][] = [];

    // Global features
    const borderTiles = Array.from(this.rlPlayer.borderTiles()).length;
    const borderPressure = this.computeBorderPressure();
    const timeAlive = this.currentTick;
    const nearestThreat = this.computeNearestThreat();
    const territoryChange = this.computeTerritoryChange(territoryPct);

    // NEW: Detect territory clusters
    const clusters = this.detectTerritoryClusters();

    // NEW: Count units by type
    const citiesCount = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'City').length;
    const portsCount = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'Port').length;
    const silosCount = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'Missile Silo').length;
    const samLaunchersCount = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'SAM Launcher').length;
    const defensePostsCount = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'Defense Post').length;
    const factoriesCount = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'Factory').length;

    // Count available nukes (units that are ready to launch)
    const atomBombsAvailable = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'Atom Bomb' && !u.tile()).length;
    const hydrogenBombsAvailable = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'Hydrogen Bomb' && !u.tile()).length;

    // Check building capabilities (simplified - just check gold)
    const CITY_COST = 5000;
    const PORT_COST = 10000;
    const SILO_COST = 15000;
    const SAM_COST = 8000;
    const canBuildCity = gold >= CITY_COST;
    const canBuildPort = gold >= PORT_COST;
    const canBuildSilo = gold >= SILO_COST;
    const canBuildSam = gold >= SAM_COST;
    const canLaunchNuke = (atomBombsAvailable > 0 || hydrogenBombsAvailable > 0) && silosCount > 0;

    // NEW: Extract unit positions for spatial observations
    const ourCitiesPositions: Array<{x: number, y: number}> = [];
    const ourPortsPositions: Array<{x: number, y: number}> = [];
    const ourSilosPositions: Array<{x: number, y: number}> = [];
    const ourSamPositions: Array<{x: number, y: number}> = [];
    const ourDefensePositions: Array<{x: number, y: number}> = [];
    const ourFactoriesPositions: Array<{x: number, y: number}> = [];

    for (const unit of this.rlPlayer.units()) {
      const tile = unit.tile();
      if (!tile) continue; // Skip units without position

      const x = this.game.x(tile);
      const y = this.game.y(tile);
      const unitType = unit.type();

      if (unitType === 'City') {
        ourCitiesPositions.push({x, y});
      } else if (unitType === 'Port') {
        ourPortsPositions.push({x, y});
      } else if (unitType === 'Missile Silo') {
        ourSilosPositions.push({x, y});
      } else if (unitType === 'SAM Launcher') {
        ourSamPositions.push({x, y});
      } else if (unitType === 'Defense Post') {
        ourDefensePositions.push({x, y});
      } else if (unitType === 'Factory') {
        ourFactoriesPositions.push({x, y});
      }
    }

    // Extract enemy building positions
    const enemyBuildingsPositions: Array<{x: number, y: number}> = [];
    for (const aiPlayer of this.aiPlayers) {
      if (!aiPlayer.isAlive()) continue;

      for (const unit of aiPlayer.units()) {
        const tile = unit.tile();
        if (!tile) continue;

        const unitType = unit.type();
        // Only track buildings, not troops
        if (['City', 'Port', 'Missile Silo', 'SAM Launcher', 'Defense Post', 'Factory'].includes(unitType)) {
          const x = this.game.x(tile);
          const y = this.game.y(tile);
          enemyBuildingsPositions.push({x, y});
        }
      }
    }

    // Update history
    this.previousTiles = tilesOwned;

    // Check game over
    // SKAI: 80% OF LAND (reachable), not 80% of the rectangle (impossible at sea).
    const has_won = territoryPct >= this.winThreshold;
    const has_lost = !this.rlPlayer.isAlive();
    const game_over = has_won || has_lost || this.currentTick >= this.maxTicks;

    return {
      tick: this.currentTick,
      game_over,
      has_won,
      has_lost,
      tiles_owned: tilesOwned,
      total_tiles: totalTiles,
      land_tiles: this.landTiles,
      territory_pct: territoryPct,
      territory_pct_all: territoryPctAll,
      water_mask: this.waterMask,
      neutral_tiles: neutralTiles,
      population,
      max_population: maxPopulation,
      population_growth_rate: populationGrowthRate,
      gold,
      num_cities: numCities,
      rank,
      total_players: this.numPlayers,
      alive_players: alivePlayers.length,
      territory_map: territoryMap,
      troop_map: troopMap,
      border_tiles: borderTiles,
      border_pressure: borderPressure,
      time_alive: timeAlive,
      nearest_threat_distance: nearestThreat,
      territory_change: territoryChange,
      clusters: clusters,  // Territory clusters
      // NEW: Economic & Military
      cities_count: citiesCount,
      ports_count: portsCount,
      silos_count: silosCount,
      sam_launchers_count: samLaunchersCount,
      defense_posts_count: defensePostsCount,
      factories_count: factoriesCount,
      atom_bombs_available: atomBombsAvailable,
      hydrogen_bombs_available: hydrogenBombsAvailable,
      can_build_city: canBuildCity,
      can_build_port: canBuildPort,
      can_build_silo: canBuildSilo,
      can_build_sam: canBuildSam,
      can_launch_nuke: canLaunchNuke,
      // NEW: Unit positions for spatial observations
      our_cities_positions: ourCitiesPositions,
      our_ports_positions: ourPortsPositions,
      our_silos_positions: ourSilosPositions,
      our_sam_positions: ourSamPositions,
      our_defense_positions: ourDefensePositions,
      our_factories_positions: ourFactoriesPositions,
      enemy_buildings_positions: enemyBuildingsPositions
    };
  }

  /**
   * Execute attack from specific cluster in direction with intensity
   * NEW: Cluster-aware attack system
   */
  attackDirection(
    direction: number,
    intensity: number,
    clusterId: number = 0,
    useSourceTile: boolean = true,
  ): boolean {
    if (!this.game || !this.rlPlayer) {
      return false;
    }

    // Direction 8 = WAIT. SKAI: waiting also releases the troops locked in flight,
    // otherwise a previously ordered attack keeps draining the population forever.
    if (direction === 8) {
      this.cancelAttacks();
      return true;
    }

    const clusters = this.detectTerritoryClusters();
    if (clusters.length === 0) {
      this.log(`WARNING: no cluster left (agent has ${this.rlPlayer.numTilesOwned()} tiles)`);
      return false;
    }
    if (clusterId < 0 || clusterId >= clusters.length) {
      clusterId = 0;
    }
    const cluster = clusters[clusterId];

    // Direction vectors: N, NE, E, SE, S, SW, W, NW
    const directionVectors = [
      { dx: 0, dy: -1 },   // 0: N
      { dx: 1, dy: -1 },   // 1: NE
      { dx: 1, dy: 0 },    // 2: E
      { dx: 1, dy: 1 },    // 3: SE
      { dx: 0, dy: 1 },    // 4: S
      { dx: -1, dy: 1 },   // 5: SW
      { dx: -1, dy: 0 },   // 6: W
      { dx: -1, dy: -1 },  // 7: NW
    ];
    const targetDir = directionVectors[direction];
    if (!targetDir) return false;

    // SKAI: troops are drawn from what the player actually holds right now.
    // The old code used a "cluster troop count" that was just the global troop
    // pool prorated by area, and it re-committed half of it on EVERY tick.
    const nOut = this.rlPlayer.outgoingAttacks().length;
    const reserve = Math.max(0, Math.floor(this.rlPlayer.troops()));
    const frac = Math.min(Math.max(intensity, 0), 1);
    const attackTroops = Math.floor(reserve * frac);
    if (this.verbose) this.log(`probing dir=${direction} reserve=${reserve} troops=${attackTroops} out_attacks=${nOut} ticks=${this.game.ticks()}`);
    if (attackTroops < 1) {
      return false;
    }

    // Choose the border tile of this cluster that faces the requested direction and
    // has the most conquestable neighbours: that is the front the attack can hold.
    // SKAI: canAttack() runs a BFS over all unowned land within Manhattan 200 for a
    // neutral tile - calling it for every border tile made each attack command cost
    // O(border x 40k tiles) on a 500x500 map (measured: 6 ticks/s with attacks vs
    // 161 ticks/s without). So: score every candidate first (pure array reads), then
    // validate only the best few.
    type Cand = { tile: number; score: number; ownerId: number; neutral: boolean };
    const cands: Cand[] = [];
    let reject = "none";
    const neutralId = this.game.terraNullius().id();

    for (const borderTile of cluster.border_tiles) {
      const x = this.game.x(borderTile);
      const y = this.game.y(borderTile);
      const targetX = x + targetDir.dx;
      const targetY = y + targetDir.dy;
      if (targetX < 0 || targetX >= this.mapWidth || targetY < 0 || targetY >= this.mapHeight) {
        reject = "offmap";
        continue;
      }
      const refIdx = targetY * this.mapWidth + targetX;
      // SKAI: the engine never pushes troops across water (AttackExecution.addNeighbors
      // skips isWater neighbours), so a water neighbour is an illegal target, not neutral land.
      if (!this.landBits[refIdx]) {
        reject = "water";
        continue;
      }
      const oid = this.game.ownerID(refIdx);
      if (oid === this.rlPlayer.smallID()) {
        reject = "own";
        continue;
      }
      let score = 0;
      if (x > 0 && this.landBits[borderTile - 1]) score++;
      if (x < this.mapWidth - 1 && this.landBits[borderTile + 1]) score++;
      if (y > 0 && this.landBits[borderTile - this.mapWidth]) score++;
      if (y < this.mapHeight - 1 && this.landBits[borderTile + this.mapWidth]) score++;
      cands.push({ tile: borderTile, score, ownerId: oid, neutral: oid === 0 });
    }

    cands.sort((a, b) => b.score - a.score);

    let bestTile: number | null = null;
    // SKAI: terraNullius().id() is `null` in this engine, so "no target found" must
    // be tracked separately from the (legitimately null) neutral target id.
    let bestTargetId: number | null = null;
    let foundTarget = false;

    for (const c of cands.slice(0, 8)) {
      if (c.neutral) {
        // Adjacent to our territory and unowned: legal unless still in the spawn phase.
        if (this.game.inSpawnPhase() || this.game.config().numSpawnPhaseTurns() +
            this.game.config().spawnImmunityDuration() > this.game.ticks()) {
          reject = "immunity";
          continue;
        }
      } else {
        const target = this.game.playerBySmallID(c.ownerId) as Player;
        if (!target.isPlayer()) continue;
        // SKAI: never attack an ally (the engine would only break the alliance for nothing)
        if (this.rlPlayer.isAlliedWith(target)) { reject = "ally"; continue; }
        if (!this.rlPlayer.canAttack(this.game.ref(c.tile % this.mapWidth + targetDir.dx, Math.floor(c.tile / this.mapWidth) + targetDir.dy))) {
          reject = "canAttack";
          continue;
        }
        bestTargetId = target.id();
      }
      bestTile = c.tile;
      if (bestTargetId === null) bestTargetId = neutralId;
      foundTarget = true;
      break;
    }

    if (!foundTarget || bestTile === null) {
      if (this.verbose) {
        this.log(`attack refused: dir=${direction} border=${cluster.border_tiles.length} reason=${reject}`);
      }
      return false;
    }

    // SKAI: never pile a second attack onto a target we are already attacking.
    // The engine merges same-target attacks, so re-issuing each tick only moved
    // troops into an ever-growing attack that conquered nothing more
    // (measured: 0.5% territory, rank 6/6, 120 000 troops locked in flight).
    // SKAI (ratio-lab): `allow_multi_front` disables this guard for the ratio sweep
    // bot, which legitimately explores several simultaneous fronts vs terra nullius.
    const active = this.rlPlayer.outgoingAttacks().filter((a) => a.isActive());
    // SKAI: at most a handful of concurrent fronts, like a human micromanaging.
    if (active.length >= this.maxAttacks) {
      if (this.verbose) this.log(`attack refused: ${active.length} attacks already active`);
      return false;
    }
    // SKAI: reinforcing a front is allowed, re-charging it every tick is not.
    if (!this.allowMultiFront) {
      for (const at of active) {
        if (at.target().id() === bestTargetId && at.troops() >= attackTroops) {
          if (this.verbose) this.log(`attack refused: front already carries ${at.troops()} troops vs offer ${attackTroops}`);
          return false;
        }
      }
    }

    const attack = new AttackExecution(
      attackTroops,
      this.rlPlayer,
      bestTargetId,
      // SKAI: the crux. With sourceTile = null the engine conquers from EVERY border
      // tile of the player (refreshToConquer), which silently turns the chosen
      // direction into a no-op. Passing the source tile confines the attack to that
      // front, which is what makes the 9-direction action space meaningful at all.
      useSourceTile ? bestTile : null,
    );
    this.game.addExecution(attack);
    if (this.verbose) {
      this.log(`attack: dir=${direction} troops=${attackTroops} from (${this.game.x(bestTile)},${this.game.y(bestTile)}) -> target ${bestTargetId}`);
    }
    return true;
  }

  /**
   * SKAI: order every outgoing attack to retreat. The engine charges a 25% loss on
   * recall, so this is a costly but real "undo" for the agent.
   */
  cancelAttacks(): number {
    if (!this.game || !this.rlPlayer) return 0;
    const attacks = this.rlPlayer.outgoingAttacks();
    for (const at of attacks) {
      if (!at.isActive()) continue;
      at.orderRetreat();
      at.executeRetreat();
    }
    return attacks.length;
  }

  /**
   * Build a unit (City, Port, Silo, SAM Launcher, Defense Post, Factory)
   * Coordinates are normalized (0-1), we convert to tile positions
   */
  buildUnit(unitType: string, tileX: number, tileY: number): boolean {
    if (!this.game || !this.rlPlayer) {
      return false;
    }

    // Convert normalized coordinates to actual tile positions
    const x = Math.floor(tileX * this.mapWidth);
    const y = Math.floor(tileY * this.mapHeight);

    // Clamp to valid range
    const clampedX = Math.max(0, Math.min(this.mapWidth - 1, x));
    const clampedY = Math.max(0, Math.min(this.mapHeight - 1, y));

    const targetTile = this.game.ref(clampedX, clampedY);

    // Check if tile is owned by RL player
    const owner = this.game.owner(targetTile);
    if (!owner.isPlayer() || (owner as Player).id() !== this.rlPlayer.id()) {
      this.log(`Cannot build ${unitType} at (${clampedX}, ${clampedY}): not owned by RL player`);
      return false;
    }

    // Check if we can build this unit type at this location
    const canBuild = this.rlPlayer.canBuild(unitType as any, targetTile);
    if (!canBuild) {
      this.log(`Cannot build ${unitType} at (${clampedX}, ${clampedY}): canBuild returned false`);
      return false;
    }

    try {
      // Build the unit
      const buildTile = canBuild === true ? targetTile : canBuild;
      this.rlPlayer.buildUnit(unitType as any, buildTile, {});

      this.log(`Built ${unitType} at (${clampedX}, ${clampedY})`);
      return true;
    } catch (error: any) {
      this.log(`Failed to build ${unitType}: ${error.message}`);
      return false;
    }
  }

  /**
   * Launch a nuclear weapon (Atom Bomb or Hydrogen Bomb)
   * Coordinates are normalized (0-1), we convert to tile positions
   */
  launchNuke(nukeType: string, targetX: number, targetY: number): boolean {
    if (!this.game || !this.rlPlayer) {
      return false;
    }

    // Convert normalized coordinates to actual tile positions
    const x = Math.floor(targetX * this.mapWidth);
    const y = Math.floor(targetY * this.mapHeight);

    // Clamp to valid range
    const clampedX = Math.max(0, Math.min(this.mapWidth - 1, x));
    const clampedY = Math.max(0, Math.min(this.mapHeight - 1, y));

    const targetTile = this.game.ref(clampedX, clampedY);

    // Find available nuke
    const availableNukes = Array.from(this.rlPlayer.units()).filter(
      u => u.type() === nukeType && !u.tile()
    );

    if (availableNukes.length === 0) {
      this.log(`No available ${nukeType} to launch`);
      return false;
    }

    // Check if we have a silo
    const silos = Array.from(this.rlPlayer.units()).filter(u => u.type() === 'Missile Silo');
    if (silos.length === 0) {
      this.log(`Cannot launch nuke: no Missile Silo available`);
      return false;
    }

    try {
      // Get the first available nuke
      const nuke = availableNukes[0];

      // Launch the nuke by placing it on the target tile
      // Note: The game engine should handle the explosion mechanics
      nuke.setTile(targetTile);

      this.log(`Launched ${nukeType} to (${clampedX}, ${clampedY})`);
      return true;
    } catch (error: any) {
      this.log(`Failed to launch ${nukeType}: ${error.message}`);
      return false;
    }
  }

  /**
   * SKAI: one pass over the map -> territory map + neutral-land count + player centroids.
   * Replaces three separate O(w*h) scans that used to run every tick.
   */
  private scanMap(): {
    map: number[][];
    own: Uint8Array;
    neutralLand: number;
    centers: Map<number, { x: number; y: number; n: number }>;
  } {
    const out: number[][] = [];
    const centers = new Map<number, { x: number; y: number; n: number }>();
    const own = new Uint8Array(this.mapWidth * this.mapHeight);
    let neutralLand = 0;

    if (!this.game || !this.rlPlayer) {
      return { map: out, own, neutralLand: 0, centers };
    }
    const game = this.game;
    const rlId = this.rlPlayer.id();
    // SKAI: numeric ids only (ownerID / smallID), no object lookups per tile
    const ownSmall = (this.rlPlayer as any).smallID();
    const aiSmall = new Map<number, number>();
    this.aiPlayers.forEach((ai, i) => aiSmall.set((ai as any).smallID(), i + 2));
    const w = this.mapWidth, h = this.mapHeight;
    const land = this.landBits;
    let ref = 0;

    for (let y = 0; y < h; y++) {
      const row: number[] = new Array(w);
      for (let x = 0; x < w; x++, ref++) {
        const oid = game.ownerID(ref);
        if (oid === 0) {
          row[x] = 0;
          if (land[ref]) neutralLand++;
        } else if (oid === ownSmall) {
          row[x] = 1;
          own[ref] = 1;
          let c = centers.get(rlId);
          if (!c) { c = { x: 0, y: 0, n: 0 }; centers.set(rlId, c); }
          c.x += x; c.y += y; c.n++;
        } else {
          row[x] = aiSmall.has(oid) ? (aiSmall.get(oid) as number) : 999;
        }
      }
      out.push(row);
    }
    return { map: out, own, neutralLand, centers };
  }

  /**
   * SKAI: downsampled land fraction (size x size). The RL observation must tell ocean
   * apart from conquerable neutral land: the engine refuses attacks across water
   * (isWater in AttackExecution.addNeighbors), so conflating them makes the map unreadable.
   */
  private buildWaterMask(gameMap: any, w: number, h: number, size: number): number[][] {
    const mask: number[][] = [];
    for (let my = 0; my < size; my++) {
      const y0 = Math.floor((my * h) / size);
      const y1 = Math.max(y0 + 1, Math.floor(((my + 1) * h) / size));
      const row: number[] = [];
      for (let mx = 0; mx < size; mx++) {
        const x0 = Math.floor((mx * w) / size);
        const x1 = Math.max(x0 + 1, Math.floor(((mx + 1) * w) / size));
        let land = 0, tot = 0;
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            tot++;
            if (gameMap.isLand(gameMap.ref(x, y))) land++;
          }
        }
        row.push(tot > 0 ? Number((land / tot).toFixed(3)) : 0);
      }
      mask.push(row);
    }
    return mask;
  }

  /**
   * Extract territory map (player IDs per tile)
   */
  private extractTerritoryMap(): number[][] {
    if (!this.game || !this.rlPlayer) {
      return [];
    }

    const map: number[][] = [];

    for (let y = 0; y < this.mapHeight; y++) {
      const row: number[] = [];
      for (let x = 0; x < this.mapWidth; x++) {
        const tile = this.game.ref(x, y);
        const owner = this.game.owner(tile);

        if (!owner.isPlayer()) {
          row.push(0);  // Neutral/Terra Nullius
        } else {
          const ownerPlayer = owner as Player;
          if (ownerPlayer.id() === this.rlPlayer!.id()) {
            row.push(1);  // RL player
          } else {
            // Find AI player index
            const aiIndex = this.aiPlayers.findIndex(ai => ai.id() === ownerPlayer.id());
            row.push(aiIndex >= 0 ? aiIndex + 2 : 999);  // AI players: 2, 3, 4, ...
          }
        }
      }
      map.push(row);
    }

    return map;
  }

  /**
   * Compute border pressure (how many enemy troops near borders)
   */
  private computeBorderPressure(): number {
    if (!this.game || !this.rlPlayer) return 0;

    const borderTiles = Array.from(this.rlPlayer.borderTiles());
    let pressure = 0;

    for (const borderTile of borderTiles) {
      const x = this.game.x(borderTile);
      const y = this.game.y(borderTile);

      // Check adjacent tiles for enemies
      const offsets = [
        { dx: -1, dy: 0 }, { dx: 1, dy: 0 },
        { dx: 0, dy: -1 }, { dx: 0, dy: 1 }
      ];

      for (const offset of offsets) {
        const nx = x + offset.dx;
        const ny = y + offset.dy;

        if (nx < 0 || nx >= this.mapWidth || ny < 0 || ny >= this.mapHeight) continue;

        const tile = this.game.ref(nx, ny);
        const owner = this.game.owner(tile);

        if (owner.isPlayer() && (owner as Player).id() !== this.rlPlayer.id()) {
          pressure += 1;
        }
      }
    }

    return pressure / Math.max(borderTiles.length, 1);
  }

  /**
   * Compute nearest threat distance
   */
  private computeNearestThreat(): number {
    if (!this.game || !this.rlPlayer) return 999;

    // SKAI: uses the centroids produced by the fused scan instead of rescanning the
    // whole map once per enemy (that was O(w*h*numEnemies) per tick).
    const me = this.centers.get(this.rlPlayer.id());
    if (!me || me.n === 0) return 999;
    const cx = me.x / me.n, cy = me.y / me.n;

    let minDistance = 999;
    for (const ai of this.aiPlayers) {
      if (!ai.isAlive()) continue;
      const c = this.centers.get(ai.id());
      if (!c || c.n === 0) continue;
      const ex = c.x / c.n, ey = c.y / c.n;
      minDistance = Math.min(minDistance, Math.hypot(cx - ex, cy - ey));
    }
    return minDistance;
  }

  /**
   * Compute territory change over last few ticks
   */
  private computeTerritoryChange(currentPct: number): number {
    this.territoryHistory.push(currentPct);
    if (this.territoryHistory.length > 10) {
      this.territoryHistory.shift();
    }

    if (this.territoryHistory.length < 2) return 0;

    const oldPct = this.territoryHistory[0];
    return currentPct - oldPct;
  }

  /**
   * Detect disconnected territory clusters using flood fill.
   * Returns up to 5 largest clusters sorted by size.
   */
  private detectTerritoryClusters(): TerritoryCluster[] {
    if (!this.game || !this.rlPlayer) return [];
    if (this.clustersCacheTick === this.currentTick) return this.clustersCache;

    // SKAI: connected components computed on the own-territory bitmap produced by the
    // fused scan. The previous implementation re-created `new Set(allTiles)` for every
    // cluster and used queue.shift() (O(n) per pop), which made it the dominant cost
    // per tick on 500x500 maps (measured 6-9 ticks/s).
    const w = this.mapWidth, h = this.mapHeight;
    const own = this.scanOwn;
    const label = new Int32Array(w * h).fill(-1);
    const queue = new Int32Array(w * h);
    const sizes: number[] = [];
    const borders: number[][] = [];
    const sumsX: number[] = [];
    const sumsY: number[] = [];

    for (let start = 0; start < own.length; start++) {
      if (!own[start] || label[start] !== -1) continue;
      const id = sizes.length;
      let head = 0, tail = 0, n = 0, sx = 0, sy = 0;
      queue[tail++] = start;
      label[start] = id;
      const borderList: number[] = [];
      while (head < tail) {
        const t = queue[head++];
        const x = t % w, y = (t - x) / w;
        n++; sx += x; sy += y;
        let isBorder = false;
        // 4-neighbours, inlined (no object allocation per neighbour)
        if (x > 0) {
          const l = t - 1;
          if (own[l]) { if (label[l] === -1) { label[l] = id; queue[tail++] = l; } } else isBorder = true;
        } else isBorder = true;
        if (x < w - 1) {
          const r = t + 1;
          if (own[r]) { if (label[r] === -1) { label[r] = id; queue[tail++] = r; } } else isBorder = true;
        } else isBorder = true;
        if (y > 0) {
          const u = t - w;
          if (own[u]) { if (label[u] === -1) { label[u] = id; queue[tail++] = u; } } else isBorder = true;
        } else isBorder = true;
        if (y < h - 1) {
          const d = t + w;
          if (own[d]) { if (label[d] === -1) { label[d] = id; queue[tail++] = d; } } else isBorder = true;
        } else isBorder = true;
        if (isBorder) borderList.push(t);
      }
      sizes.push(n);
      borders.push(borderList);
      sumsX.push(sx); sumsY.push(sy);
    }

    const totalTiles = sizes.reduce((a, b) => a + b, 0);
    const totalTroops = this.rlPlayer.troops();
    const order = sizes.map((_, i) => i).sort((a, b) => sizes[b] - sizes[a]).slice(0, 5);
    const clusters: TerritoryCluster[] = order.map((i, idx) => ({
      id: idx,
      // SKAI: the index list is no longer shipped (it cost ~40 kB of JSON per tick for
      // a number nobody reads); consumers use tile_count instead.
      tiles: [] as number[],
      tile_count: sizes[i],
      border_tiles: borders[i],
      center_x: sizes[i] ? sumsX[i] / sizes[i] : 0,
      center_y: sizes[i] ? sumsY[i] / sizes[i] : 0,
      troop_count: totalTiles > 0 ? Math.floor((totalTroops * sizes[i]) / totalTiles) : 0,
    }));

    this.clustersCacheTick = this.currentTick;
    this.clustersCache = clusters;
    return clusters;
  }

  /**
   * Check if a tile is on the border of our territory.
   */
  private isBorderTile(tile: number): boolean {
    if (!this.game || !this.rlPlayer) return false;

    const x = this.game.x(tile);
    const y = this.game.y(tile);

    const neighbors = [
      { dx: 0, dy: -1 },
      { dx: 1, dy: 0 },
      { dx: 0, dy: 1 },
      { dx: -1, dy: 0 }
    ];

    for (const { dx, dy } of neighbors) {
      const nx = x + dx;
      const ny = y + dy;

      if (nx >= 0 && nx < this.mapWidth && ny >= 0 && ny < this.mapHeight) {
        const neighborTile = this.game.ref(nx, ny);
        const owner = this.game.owner(neighborTile);

        // If neighbor is not ours, we're on a border
        if (!owner.isPlayer() || (owner as Player).id() !== this.rlPlayer.id()) {
          return true;
        }
      }
    }

    return false;
  }

  private log(message: string): void {
    console.error(`[GameBridge] ${message}`);
  }

  setMaxAttacks(n: number) {
    this.maxAttacks = Math.max(1, Math.min(12, Math.floor(n)));
  }

  setAllowMultiFront(v: boolean) {
    this.allowMultiFront = !!v;
  }

  setGameConfigMode(m: 'default' | 'test') {
    this.gameConfigMode = m === 'test' ? 'test' : 'default';
  }

  setVerbose(v: boolean) {
    this.verbose = v;
  }

  setWinThreshold(t: number) {
    this.winThreshold = Math.min(Math.max(t, 0.05), 1.0);
  }

  close() {
    this.game = null;
    this.rlPlayer = null;
    this.aiPlayers = [];
  }
}

// Main IPC loop
async function main() {
  const bridge = new GameBridge();
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
  });

  console.error('[GameBridge] Ready for commands');

  rl.on('line', async (line: string) => {
    try {
      const command: Command = JSON.parse(line);
      let response: any = { success: true };

      switch (command.type) {
        case 'reset':
          if (command.win_threshold !== undefined) {
            bridge.setWinThreshold(command.win_threshold);
          }
          bridge.setVerbose(command.verbose ?? false);
          if (command.game_config) bridge.setGameConfigMode(command.game_config as any);
          if (command.max_attacks !== undefined) bridge.setMaxAttacks(command.max_attacks);
          if (command.allow_multi_front !== undefined) bridge.setAllowMultiFront(command.allow_multi_front);
          if (command.spawn_mode) bridge.spawnMode = command.spawn_mode === 'ring' ? 'ring' : 'land';
          if (command.spawn_seed !== undefined) bridge.spawnSeed = command.spawn_seed >>> 0;
          if (command.obs_size !== undefined) bridge.obsSize = Math.max(0, Math.floor(command.obs_size));
          response.state = await bridge.initialize(
            command.map_name || 'plains',
            command.num_players || 11
          );
          break;

        case 'tick':
          response.state = bridge.tick();
          break;

        case 'get_state':
          response.state = bridge.getState();
          break;

        case 'attack_direction':
          response.success = bridge.attackDirection(
            command.direction ?? 8,
            command.intensity ?? 0.5,
            command.cluster_id ?? 0,
            command.source_tile ?? true,
          );
          break;

        case 'cancel_attacks':
          response.cancelled = bridge.cancelAttacks();
          break;

        case 'build_unit':
          if (!command.unit_type || command.tile_x === undefined || command.tile_y === undefined) {
            response.success = false;
            response.error = 'Missing parameters for build_unit command';
          } else {
            response.success = bridge.buildUnit(
              command.unit_type,
              command.tile_x,
              command.tile_y
            );
          }
          break;

        case 'launch_nuke':
          if (!command.nuke_type || command.tile_x === undefined || command.tile_y === undefined) {
            response.success = false;
            response.error = 'Missing parameters for launch_nuke command';
          } else {
            response.success = bridge.launchNuke(
              command.nuke_type,
              command.tile_x,
              command.tile_y
            );
          }
          break;

        case 'shutdown':
          bridge.close();
          console.log(JSON.stringify(response));
          process.exit(0);
          break;

        default:
          response.success = false;
          response.error = `Unknown command type: ${command.type}`;
      }

      console.log(JSON.stringify(response));
    } catch (error: any) {
      console.log(JSON.stringify({
        success: false,
        error: error.message || String(error)
      }));
    }
  });

  rl.on('close', () => {
    bridge.close();
    process.exit(0);
  });
}

main().catch(error => {
  console.error('[GameBridge FATAL]', error);
  process.exit(1);
});
