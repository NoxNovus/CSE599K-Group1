import torch
import os
import tqdm
import safetensors

########################################
# WeightManager: Loading and Managing Weights
########################################

class WeightManager:
    """
    Manages loading and processing model weights from safetensors files.
    """
    
    @staticmethod
    def load_tensors(tensor_path: str) -> dict:
        """
        Loads all tensors from safetensors files in a directory.

        Args:
            tensor_path (str): Path to directory with safetensors files.
            
        Returns:
            dict: Mapping from tensor names to torch.Tensor objects.
        """
        original_tensors = {}
        # Iterate through files in the directory
        for file in tqdm.tqdm(os.listdir(tensor_path), desc="Loading safetensors"):
            if file.endswith(".safetensors"):
                # Open file in PyTorch mode
                tensors = safetensors.safe_open(os.path.join(tensor_path, file), 'pt')
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name)
                    original_tensors[name] = tensor
        return original_tensors

    def __init__(self):
        self.weight_map = {}

    def load_from_safe_tensor(self, tensor_path: str) -> None:
        """
        Loads weights from safetensors files, converts them to fp16, and moves them to GPU.

        Args:
            tensor_path (str): Path to directory with safetensors files.
        """
        self.weight_map = WeightManager.load_tensors(tensor_path)
        # Convert weights to half precision and move to CUDA device
        for key in self.weight_map.keys():
            self.weight_map[key] = self.weight_map[key].half().to('cuda')

    def set_weight(self, operation_list, total_layers: int) -> None:
        """
        Applies processing operations on weights.

        Args:
            operation_list (list): List of operations, each having a processWeight() method.
            total_layers (int): Total number of transformer layers.
        """
        for op in operation_list:
            op.processWeight(self.weight_map, total_layers)

########################################
# Rotary Positional Embedding (RoPE) Functions
########################################

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the last half of the tensor along the final dimension.
    
    Args:
        x (torch.Tensor): Input tensor with shape [..., 2*d_half].
        
    Returns:
        torch.Tensor: Tensor with rotated halves.
    """
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]  # First half of features
    x2 = x[..., dim // 2:]   # Second half of features
    return torch.cat([-x2, x1], dim=-1)

def apply_batched_rope(x: torch.Tensor, output: torch.Tensor, head_dim: int, offset: int = 0) -> None:
    batch_size, seq_len, hidden_dim = x.shape
    device = x.device
    dtype = x.dtype
    num_heads = hidden_dim // head_dim
    # Create positions: shape [seq_len]
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)

    base = 500000.0
    # Compute inverse frequency: shape [head_dim/2]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))

    # Compute frequency embeddings
    with torch.autocast(device_type=device.type, enabled=False):
        freqs = torch.outer(positions, inv_freq) 
        # Duplicate frequencies for cos and sin parts:
        emb = torch.cat((freqs, freqs), dim=-1) 
        cos = emb.cos()                         
        sin = emb.sin()                       

    # Reshape for multi-head compatibility:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Reshape x for applying RoPE:
    x = x.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    # Apply RoPE rotation
    x_rotated = x * cos + rotate_half(x) * sin
    # Restore original shape and copy into outpu
    output.copy_(x_rotated.transpose(1, 2).reshape(batch_size, seq_len, -1).to(dtype=dtype))

def apply_rope(x: torch.Tensor, output: torch.Tensor, head_dim: int, offset: int = 0) -> None:
    """
    Applies RoPE (Rotary Positional Embedding) to the input tensor.
    
    RoPE adds position-dependent rotations to the tensor.
    
    Args:
        x (torch.Tensor): Input tensor with shape [seq_len, head_dim].
        output (torch.Tensor): Tensor to store the result (same shape as x).
        head_dim (int): Dimensionality of each attention head.
        offset (int): Positional offset.
    """
    seq_len, _ = x.shape  # [seq_len, hidden_dim]
    device = x.device
    dtype = x.dtype

    # Create positions: shape [seq_len]
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)

    base = 500000.0
    # Compute inverse frequency: shape [head_dim/2]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64)
                                 .float().to(device) / head_dim))
    
    # Expand dimensions for broadcasting:
    inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)
    position_ids_expanded = positions[None, None, :].float()
    
    # Compute frequency embeddings
    with torch.autocast(device_type=device.type, enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        # Duplicate frequencies for cos and sin parts:
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    
    # Reshape for multi-head compatibility:
    cos = cos.unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [1, 1, seq_len, head_dim]

    # Reshape x for applying RoPE:
    x = x.reshape(1, seq_len, -1, head_dim).transpose(1, 2)
    # Apply RoPE rotation
    x_rotated = x * cos + rotate_half(x) * sin
    # Restore original shape and copy into output
    output.copy_(x_rotated.transpose(1, 2).reshape(seq_len, -1).to(dtype=dtype))

########################################
# Model Weight Extraction
########################################

def extract_model_weights(weight_map: dict, layers: int) -> dict:
    """
    Extracts and organizes model weights from a weight_map into a dictionary.

    Args:
        weight_map (dict): Dictionary containing the full mapping of weight tensors.
        layers (int): Total number of transformer layers.

    Returns:
        dict: Dictionary with keys for embedding, layer-specific weights, and final projection weights.
    """
    weights = {}
    weights["embedding"] = weight_map["model.embed_tokens.weight"]
    weights["layernormAttn_weight"] = [
        weight_map[f"model.layers.{layer}.input_layernorm.weight"] for layer in range(layers)
    ]
    weights["self_attn_k_proj_weight"] = [
        weight_map[f"model.layers.{layer}.self_attn.k_proj.weight"] for layer in range(layers)
    ]
    weights["self_attn_v_proj_weight"] = [
        weight_map[f"model.layers.{layer}.self_attn.v_proj.weight"] for layer in range(layers)
    ]
    weights["self_attn_q_proj_weight"] = [
        weight_map[f"model.layers.{layer}.self_attn.q_proj.weight"] for layer in range(layers)
    ]
    weights["o_proj_weight"] = [
        weight_map[f"model.layers.{layer}.self_attn.o_proj.weight"] for layer in range(layers)
    ]
    weights["layernormFFN_weight"] = [
        weight_map[f"model.layers.{layer}.post_attention_layernorm.weight"] for layer in range(layers)
    ]
    weights["up_proj_weight"] = [
        weight_map[f"model.layers.{layer}.mlp.up_proj.weight"] for layer in range(layers)
    ]
    weights["gate_proj_weight"] = [
        weight_map[f"model.layers.{layer}.mlp.gate_proj.weight"] for layer in range(layers)
    ]
    weights["down_proj_weight"] = [
        weight_map[f"model.layers.{layer}.mlp.down_proj.weight"] for layer in range(layers)
    ]
    weights["model_layernorm_weight"] = weight_map["model.norm.weight"]
    weights["lm_head_weight"] = weight_map["lm_head.weight"]
    return weights


prefill_input = """
High above the jagged peaks of the Emberfall Mountains, where dawn’s first light turned obsidian spires into blood-red sentinels, there lived a dragon unlike any ever known to mortal or immortal. Her name was Aurestriel, the Thousand-Scaled Flame—so named for the kaleidoscope of colors that shimmered across her hide, each scale small yet radiant, shifting through fiery crimsons, molten golds, dusk-blues, and smoky violets as if the sunrise itself had been captured and woven into her flesh.

Her lair, the Maw of Cinders, yawned at the mountain’s summit: a cavernous chamber hollowed from volcanic rock, its walls alive with veins of glowing magma. Rivers of molten rock threaded between stalactites and stalagmites, lighting the vast hall with flickering, ruby brilliance. Ancient banners, burned at the edges but still loyal to their master, tattered and singed, hung along the walls, bearing sigils of long-fallen dynasties—trophies of wars waged and won (or lost) centuries ago. Piles of jeweled ingots, silver weapons, and crowns of mythic renown lay heaped in pyres, waiting only for her whim to incinerate or keep.

Despite her fearsome visage and the terror she inspired in the folk of the lowlands, Aurestriel’s heart harbored a solitude deeper than any abyss. Millennia had passed since dragons had been more than distant myth, and every century found her the lone survivor of a dwindling lineage. Mortals no longer called upon dragons for wisdom; they only feared their power. Yet still she watched the world below with curious eyes, noting every ember of hope and every spark of ruin in equal measure.

One crisp autumn eve, as the moons Aria and Callum rose together to grace the sky with their pale luminescence, a woodman named Branik stumbled upon the mountain’s lower slopes. He was a man of simple means—his clothes rough-spun, his face lined from years of labor—but in his heart lay an audacious dream: to find the Whispering Flame, an elder dragon said to grant a single boon to those brave enough to seek her. For Branik carried a burden heavier than any axe: his only daughter, Merin, lay stricken by a wasting sickness, her life ebbing like water through a sieve. Legends whispered that a dragon’s fire, refined through ancient magic, could heal any malady. And so, with trembling hands and steadfast will, he climbed.

Aurestriel saw him long before he saw her. From her perch upon a volcanic terrace, she watched his silhouette, small yet determined, picking his way along treacherous ridges. Curiosity stirred within her—a feeling she had not indulged in countless centuries. Mortal bravery was a rare gem. When Branik finally reached the edge of her domain, he fell to one knee, breathless and awed, and raised a white flag—a scrap of linen he had found in an abandoned chapel—as a token of peace.

“Aurestriel,” he called, voice echoing in the hollow night, “I beseech you: spare me your wrath and grant me but a moment of your time. My daughter lies dying. I seek not to slay nor to steal, but to beg your mercy. A single scale, a breath of your flame—such is all I ask, to restore her life.”

Silence reigned for a heartbeat too long; then the air quivered with raw power as the dragon descended. Each beat of her wings summoned tremors; embers danced in the sky like fireflies in a storm. Branik’s knees buckled, but he held firm, refusing to show fear—an act of courage that softened the ancient wyrm’s gaze.

“Child of clay,” Aurestriel’s voice boomed, each word resonant as rolling thunder, “you come seeking the gift of life from one who breathes fire and death. Why should I grant what mortals have long since forgotten to honor?”

Branik’s heart pounded, but he answered without hesitation. “Because I ask not for gold nor land, but to save the life of one whom I love beyond all treasures. My scales and my blood are yours to claim, should you wish them in payment. But let her not perish for lack of hope.”

Aurestriel paused, her long snout lowering until her golden eyes burned inches from his own. Legends told of dragons who devoured suitors, but Branik felt no fear now—only the quiet determination that had driven him here. And within that, Aurestriel sensed something she had not felt in ages: kinship.

“Rise, Branik of the Lowlands,” she intoned. “I shall grant your plea. But know this: magic once broken cannot be unwritten, and fate is not a tapestry to be unraveled at whim. Should your daughter awaken, her life shall be bound to mine, as I bind this flame to her breath. Are you prepared for such a bond?”

He swallowed, chest rising and falling with a shudder. “If it saves her, then yes. Even if it costs me my own.”

Aurestriel’s maw parted as she exhaled a plume of flame so bright it painted the mountain with molten hues. Branik shielded his eyes, but when the light subsided, he saw her blow forth a single, coruscating scale. It hovered in the air, pulsing with incandescent warmth like a miniature sun. The dragon extended her claw, and the scale floated into Branik’s outstretched hand, its surface etched with living runes that shifted like galaxies in motion.

“Go,” she commanded. “And bring your daughter to me at dawn. Lay her body upon the Altar of Embers at the heart of this lair. There, by my breath, shall she be reborn—or die anew.”

Branik bowed, tears blurring the scale’s brilliance. “Thank you, Great Flame.”

With that, he fled down the mountain, clutching his priceless burden, while Aurestriel returned to her lofty roost, her thoughts as turbulent as the molten rivers below. What had she done? In all her long existence, she had never woven her magic with that of a mortal. The bond she set in motion would reshape both their fates.

The long night passed like a dream painted in orange and gold. As the first light of dawn crept over the horizon, Branik—sickly and exhausted—arrived at the Maw of Cinders once more, carrying Merin in his arms. Her skin was pale as dawn’s mist, her breath shallow whispers. Gently, he laid her upon a dais carved from obsidian and etched with draconic runes older than the mountains themselves.

Aurestriel lowered her great head and spoke the Word of Unmaking in a tone soft as a lullaby. From her throat escaped a breath that shone with a thousand hues—flames tempered in the dragon’s soul rather than mere fire. They billowed over the girl, enveloping her in warmth without burning, filling every fiber of her being with raw, renewing power.

Merin's chest heaved as her lungs filled with new life. Her eyes fluttered open, reflecting the spectrum of flames dancing in the cavern. Branik wept with joy and relief, scooping his daughter into his arms as she gazed up at the dragon with wonder.

But as Merin’s strength returned, Aurestriel felt a tug—a subtle but immutable pull that reached across the chasm between their essences. The magic of the scale had woven their souls together, an unbreakable tether that linked her fate to the girl’s. Should Merin ever falter, the dragon would feel the tremor in her own heart; should the dragon perish, the girl too would fade into myth.

Understanding this, Aurestriel gently touched Merin’s brow with a talon as soft as a feather. “Rise, child of hope,” she whispered. “Live well, and honor the bond we share.”

With reverent steps, Branik and Merin departed the Maw of Cinders before the mountain’s heat could overtake them, leaving the dragon alone once more. Yet she was changed. In the days that followed, she felt echoes of the girl’s laughter like ripples across her spirit. She found herself gazing toward the lowlands more often, wondering what perils Merin might face, and whether she could lend aid from afar.

Seasons turned and empires shifted. Aurestriel resumed her centennial journeys to the Dragonkin Council, carrying within her the memory of a mortal’s love—something more ancient than any bargain, more potent than any hoarded treasure. She spoke of it among her kin as a testament to the power of compassion bridging worlds. And though some elder wyrms scoffed at notions of mortal bonds, even the hardest of stone-scaled dragons felt their hearts warm at her words.

Merin grew into a woman of uncommon vitality and spirit. Gifted with a draconic spark in her veins, she apprenticed herself to the healers of the Breathwood Monastery, using her gift to mend wounds both of body and spirit, always mindful of the dragon to whom she owed her life. From time to time, she would climb the mountain to offer gifts—freshwater from the Crystal Springs, silken banners of gratitude, poems sealed in scrolls bound with gold leaf—tokens of a love that transcended species.

One midsummer’s eve, as the twin moons aligned in a rare celestial dance, a dark omen blotted out the skies: a plague of shadows, drifting across kingdoms like living night, extinguishing life with a single touch. Mortals cried for aid, but the dragons were bound by ancient oaths not to intervene directly in mortal wars. Aurestriel felt the girl’s fear as though it were her own, and Merin’s pleas reverberated in her chest like the echo of a drum.

Unable to stand idle, Aurestriel convened the Dragonkin Council in the Vale of Ancients, unveiling the shadow plague as a transgression against the Great Balance. The elder wyrms debated whether to break their centuries-old code. Lysandria, Celestial Whisper of the North, argued that all life—mortal and draconic—was intertwined, and that mercy must guide them. Vythorax, Stormforger, thundered his agreement, for even storms must yield to compassion. Only Obsidion, Stoneheart, stood firm, recalling past betrayals when dragons’ interventions birthed tyrants.

In the end, it was Aurestriel’s voice that tipped the scales. She spoke of Merin, of the unlooked-for bond that had saved a single life and rippled outwards to mend countless wounds. “If one mortal’s worth can be measured by the hope she kindles,” she rumbled, “then we must act, for our oath is to the spark of life itself.”

With ancient magic rekindled, the dragons rose as one. Aurestriel led the charge, plunging into the heart of the shadow-plague’s domain, her wings carving arcs of fire through the inky veil. At her side soared Merin—transformed by draconic gift into a being of flame-kissed grace—her laughter a clarion of defiance that shattered the shadow like glass.

Aurestriel’s wings beat back the roiling darkness as she plummeted into the heart of the shadow-plague’s domain—a realm of living night that twisted mountain and forest alike into grotesque caricatures. Trees with bark black as obsidian writhed like serpents, their branches coiling toward the sky; pools of water lay stilled beneath glassy surfaces, reflections drained of color and warmth. Every step the dragons took crackled with heat, burning away tendrils of shadow before they could snuff out life itself.

By her side, Merin—her new draconic heritage radiant in every movement—soared on feathered fire. Scales glimmered beneath her skin, like embers glowing beneath ash, and her hair streamed behind her in a mantle of gold and bronze. Though now a being of profound power, she wore no arrogance; her eyes shone with fierce empathy. With each beat of her wings, her laughter rang out—a pure bell-tone that shattered shadows into motes of dust.

Together, they cleaved a path toward the plague’s epicenter: the Ruinous Obelisk, a shattered pillar of twisted stone and black crystal that pulsed with malignant life. Around its base writhed the umbral horde—creatures born of despair, their limbs elongated, faces mask-like and devoid of recognition. Wherever they touched, they left behind rot and silence. Yet Aurestriel and Merin needed no weapons but their very presence. As scales met gloom, fire met frost, and light met oblivion, each impact dissolved a creature into a burst of glittering sparks.

High above, the other dragons circled in tandem: Lysandria the Frost-Wind, her breath a gale of diamondine shards that froze shadow-things mid-step; Vythorax the Stormforger, roaring with thunder as he hurled bolts of electrified flame; even Obsidion the Stoneheart, whose obsidian hide glowed molten at the seams, carved a path with earthen fists turned magma hammers. The sky itself burned with draconic fury.

Yet the plague was cunning. Wherever they drove it back, it seeped forth from hidden fissures, reknitting its terrible will into new forms—tendrils of living smoke, waves of pallid mist shaped like screaming faces, pillars of crawling shadow that writhed along the ground. The Ruinous Obelisk fed them, its heart a nexus of dark magic drawn from the deepest despair of mortal souls.

Aurestriel felt the pull at her core: the Obelisk’s energy thrummed through the tether to Merin, echoing in the girl’s very bones. It sought to corrupt—to twist the bond and feed upon the hope that link sustained. She beat her wings, sending up a gale that scattered drifting shadows like leaves, and roared. “To the heart!” she thundered. “We starve its power by severing it at its source!”

Merin nodded, folding into a dive that scorched the air into molten streaks. She alighted upon the cracked plinth of the Obelisk, her taloned feet sparking against ancient runes. Channeling Aurestriel’s gift, she raised both arms aloft and unleashed a breath of prismatic fire—flames laced with the warmth of compassion, the light of compassion shaped by mortal empathy. The fire clawed up the Obelisk’s sides in a cascade of rainbow tongues, melting jet and crystal into rivulets of incandescent glass.

But just as victory seemed within reach, a new horror burst from the Obelisk’s crown: the Heart of Shadow itself—a living orb of writhing darkness, like a storm-cloud spun into flesh. It pulsed with malevolent intelligence, and from it sprang catalysts of the plague—bats the size of harts, sinewy and blind, screeching with a hunger for life; worms of shadow that burrowed into flesh to gnaw at hope itself.

Aurestriel lashed out, flaming talons cleaving through battalions of fleeing bats; Lysandria froze worms mid‐swallow, cracking them with crystalline hail. But the Heart’s tendrils stretched toward Merin. Aurestriel saw her daughter‐friend falter for a breath, the bond between them singing with a note of pain. Instinct overrode thought: the great wyrm dove, shielding Merin beneath her vast wings. In that instant, one of the Heart’s tendrils coiled itself around her foreleg, sinking in like a leech. Aurestriel recoiled, roaring, as the darkness began to seep into her flesh.

Merin’s eyes widened in alarm—and sorrow. She sprang forward, channeling every ounce of her newfound power. Flames roared from her lips, not merely fire but living light: a manifestation of her own spirit united with the dragon’s. She drove it into Aurestriel’s wound, a conflagration that burned away the tendril and cauterized the corruption. The dragon howled, and the shockwave shattered more shards of the Obelisk’s grip.

“Stand, Aurestriel!” Merin called, her voice firm, unwavering. “Our bond is steel. Let them know what hope is forged of!”

Bolstered by Merin’s determination, Aurestriel leapt back into the fray with renewed vigor. Together they danced a lethal ballet of flame and flame-kissed wind, their powers intertwined like lightning coiling in a thunderstorm. With a final, cataclysmic cry, Aurestriel lashed forth a blast of draconic fire so pure and incandescent that it rended the very fabric of shadow. The Heart of Shadow imploded, shrieking as it collapsed inward, its tendrils retreating like serpents scalded by boiling water. The Obelisk cracked and fell silent, its black crystal fracturing into motes that drifted away on a warm breeze.

When the air cleared, the valley below was still for the first time in months. Sunlight, unfiltered and bright, bathed the land. Rivers flowed again with crystalline water; forests sighed as leaves fluttered free of blight; even the ruined fortresses at the edge of the plague’s reach glimmered with new promise. Mortals, emerging from their hideaways, saw the dragons wheeling overhead and wept blessings to the sky.

Aurestriel landed upon the fractured base of the Obelisk, her body scorched but her spirit unbowed. Merin alighted beside her, steps faltering only for a heartbeat before she steadied herself, smiling. The other dragons descended in a procession of fire and frost, lightning and stone, circling around their two champions. Silence fell, reverent as a hymn.

Lysandria spoke first, her voice like a thawing brook. “Sister, you have shown us that bonds of compassion can conquer despair.” Vythorax nodded, thunder rolling in his throat as he added, “Let this day mark a new dawn for us all—a time when we stand not aloof, but among those we protect.” Even Obsidion inclined his head, the molten glow in his eyes softer than any heat.

Aurestriel turned to Merin and laid a gentle claw upon her shoulder. “Child of my heart,” she rumbled, “you have proven the true power of hope. You have not merely borrowed my flame; you have carried it forth into the world. Yours is the spark that will rekindle dying embers.”

Merin bowed her head, her cheeks damp with tears of relief and triumph. “I owe you everything, Aurestriel. But it is you who taught me that life is worth the flame it burns for—and that even the smallest spark can cast out the deepest shadow.”

In the days that followed, the dragons and mortals worked side by side to heal the land. Draconic fire was used to smelt away corrupted earth; Frost-Wind carved new channels for purified rivers; Stormforger’s lightning rekindled lightning-rods in every city, protecting them from lingering dark energies. Under the guidance of Merin—now hailed as the Dragon-Bound Healer—monasteries and temples rose anew, their doctrine teaching unity rather than fear.

Aurestriel made her peace with the world she had watched for millennia. No longer would dragons dwell only in legend’s margins; they had thrown open the gates between sky and earth, siding with life itself against oblivion. When elders spoke of dragons once more, it would not be only of terror and hoarding, but of guardianship, of compassion tempered with power.

But then, Aurestriel encountered a scary dinosaour named Rich Chen!
"""